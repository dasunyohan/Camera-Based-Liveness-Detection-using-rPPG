import os
import time
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import butter, filtfilt

# Optional: ICA
try:
    from sklearn.decomposition import FastICA
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

import mediapipe as mp


@dataclass
class ROI:
    name: str
    indices: List[int]
    color_bgr: Tuple[int, int, int]


class RPPGEngine:
    def __init__(
        self,
        model_path: str,
        fs: int = 30,
        buffer_seconds: int = 12,
        update_every_frames: int = 10,
        bpm_smooth_window: int = 5,
        band_low: float = 0.7,
        band_high: float = 4.0,
        waveform_len: int = 300,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"FaceLandmarker model not found at: {model_path}\n"
                f"Download a .task file (e.g., face_landmarker.task) and place it there."
            )

        self.model_path = model_path
        self.fs = fs
        self.buffer_seconds = buffer_seconds
        self.buffer_size = int(fs * buffer_seconds)
        self.update_every_frames = update_every_frames
        self.bpm_smooth_window = bpm_smooth_window
        self.band_low = band_low
        self.band_high = band_high
        self.time_buffer = deque(maxlen=self.buffer_size)  # real timestamps (seconds)

        self.rois = [
            ROI("forehead",    [10, 338, 297, 332], (0, 255, 0)),
            ROI("left_cheek",  [234, 93, 132, 58],  (255, 0, 0)),
            ROI("right_cheek", [454, 323, 361, 288], (0, 0, 255)),
        ]

        # Buffers
        self.rgb_buffer = deque(maxlen=self.buffer_size)          # fused RGB per frame
        self.green_wave = deque(maxlen=waveform_len)              # for demo waveform
        self.bpm_history = deque(maxlen=waveform_len)             # smoothed BPM trend
        self._bpm_smooth = deque(maxlen=bpm_smooth_window)

        # State / telemetry
        self.frame_idx = 0
        self.face_detected = False
        self.current_bpm: float = 0.0
        self.last_bpm_raw: Optional[float] = None
        self.last_polygons: Dict[str, List[List[int]]] = {}
        self._fps_est: Optional[float] = None
        self._last_frame_ts: Optional[float] = None

        # IMPORTANT: FaceLandmarker VIDEO mode requires monotonically increasing timestamps.
        self._ts_ms = 0
        self._frame_ms = max(1, int(1000 / self.fs))

        # ---- MediaPipe Tasks FaceLandmarker ----
        # Uses mp.tasks.vision.FaceLandmarkerOptions + BaseOptions(model_asset_path=...)
        # per official docs. :contentReference[oaicite:1]{index=1}
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    # ---------------- Signal helpers ----------------
    def _bandpass(self, x: np.ndarray, fs: float, order: int = 3) -> np.ndarray:
        if len(x) < 10 or fs <= 0:
            return x
        nyq = 0.5 * fs
        low = self.band_low / nyq
        high = self.band_high / nyq
        # avoid invalid filter when fps is low or noisy
        low = max(low, 1e-4)
        high = min(high, 0.99)
        if not (0 < low < high < 1):
            return x
        b, a = butter(order, [low, high], btype="band")
        # filtfilt needs enough samples; otherwise just return raw
        padlen = 3 * (max(len(a), len(b)) - 1)
        if len(x) <= padlen:
            return x
        return filtfilt(b, a, x)

    def _estimate_bpm_fft(self, signal: np.ndarray, fs: float) -> Optional[float]:
        if len(signal) < max(10, int(fs * 5)) or fs <= 0:
            return None
        sig = signal.astype(np.float32)
        sig = sig - np.mean(sig)
        sig = self._bandpass(sig, fs=fs)

        freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
        fft_vals = np.abs(np.fft.rfft(sig))
        band = (freqs >= self.band_low) & (freqs <= self.band_high)
        if not np.any(band):
            return None
        peak_freq = freqs[band][np.argmax(fft_vals[band])]
        bpm = float(peak_freq * 60.0)
        if not np.isfinite(bpm) or bpm <= 0:
            return None
        return bpm
    def _signal_quality(self, signal: np.ndarray) -> float:
        """
        Simple quality score [0..1] based on peak prominence in HR band.
        """
        if len(signal) < self.fs * 5:
            return 0.0
        sig = signal.astype(np.float32)
        sig = sig - np.mean(sig)

        freqs = np.fft.rfftfreq(len(sig), d=1.0 / self.fs)
        fft_vals = np.abs(np.fft.rfft(sig))
        band = (freqs >= self.band_low) & (freqs <= self.band_high)
        if not np.any(band):
            return 0.0
        band_vals = fft_vals[band]
        peak = float(np.max(band_vals))
        med = float(np.median(band_vals) + 1e-8)
        ratio = peak / med
        # ratio ~1-2 = meh, ratio 6-10+ = strong
        return float(np.clip((ratio - 2.0) / 8.0, 0.0, 1.0))

    # ---------------- ROI extraction ----------------
    def _extract_fused_rgb(
        self,
        frame_bgr: np.ndarray,
        landmarks_norm,  # list of normalized landmarks with .x .y
    ) -> Tuple[Optional[np.ndarray], Dict[str, List[List[int]]]]:
        h, w, _ = frame_bgr.shape
        roi_rgbs = []
        polygons_out: Dict[str, List[List[int]]] = {}

        for roi in self.rois:
            pts = np.array(
                [(int(landmarks_norm[i].x * w), int(landmarks_norm[i].y * h)) for i in roi.indices],
                dtype=np.int32,
            )
            polygons_out[roi.name] = pts.tolist()

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            pixels = frame_bgr[mask == 255]
            if len(pixels) == 0:
                continue

            mean_bgr = np.mean(pixels, axis=0)
            mean_rgb = mean_bgr[::-1]  # RGB
            roi_rgbs.append(mean_rgb)

        if not roi_rgbs:
            return None, polygons_out

        fused = np.mean(roi_rgbs, axis=0).astype(np.float32)
        return fused, polygons_out

    # ---------------- Main entry ----------------
    def process_frame(self, frame_bgr: np.ndarray) -> Dict:
        self.frame_idx += 1

        # fps estimate
        now = time.time()
        self.time_buffer.append(now)
        if self._last_frame_ts is not None:
            dt = now - self._last_frame_ts
            if dt > 0:
                inst = 1.0 / dt
                self._fps_est = inst if self._fps_est is None else (0.9 * self._fps_est + 0.1 * inst)
        self._last_frame_ts = now

        # MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Monotonic timestamps (required by detect_for_video) :contentReference[oaicite:2]{index=2}
        ts_ms = int(time.monotonic() * 1000)  # real, monotonic timestamp in ms
        result = self.landmarker.detect_for_video(mp_image, ts_ms)

        self.face_detected = bool(result.face_landmarks)

        fused_rgb = None
        polygons = {}
        if self.face_detected:
            landmarks = result.face_landmarks[0]
            fused_rgb, polygons = self._extract_fused_rgb(frame_bgr, landmarks)

        self.last_polygons = polygons

        # Update buffers
        if fused_rgb is not None:
            self.rgb_buffer.append(fused_rgb)
            self.green_wave.append(float(fused_rgb[1]))
        elif len(self.rgb_buffer) > 0:
            # hold last value
            self.rgb_buffer.append(self.rgb_buffer[-1])
            self.green_wave.append(self.green_wave[-1] if self.green_wave else 0.0)
        else:
            # nothing yet
            return self.get_status()

        quality = None

        # BPM update
        if (
            self.frame_idx % self.update_every_frames == 0
            and len(self.rgb_buffer) >= int(self.fs * 5)
        ):

            # Compute effective FPS from real timestamps
            fs_eff = float(self.fs)
            if len(self.time_buffer) >= 10:
                dt = float(self.time_buffer[-1] - self.time_buffer[0])
                if dt > 0:
                    fs_eff = (len(self.time_buffer) - 1) / dt

            fs_eff = float(np.clip(fs_eff, 5.0, 60.0))

            data = np.array(self.rgb_buffer, dtype=np.float32)
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)

            component = data[:, 1]  # green channel

            # Optional ICA
            """if SKLEARN_AVAILABLE:
                try:
                    ica = FastICA(n_components=3, max_iter=500, random_state=0)
                    comps = ica.fit_transform(data)

                    powers = []
                    for i in range(3):
                        filt = self._bandpass(comps[:, i], fs=fs_eff)
                        powers.append(np.std(filt))

                    component = comps[int(np.argmax(powers))]

                except Exception:
                    pass"""

            bpm_raw = self._estimate_bpm_fft(component, fs=fs_eff)
            quality = self._signal_quality(component)

            if self.frame_idx % 50 == 0:
                print(
                    "DBG len(rgb_buffer)=", len(self.rgb_buffer),
                    "fs_nom=", self.fs,
                    "fs_eff=", round(fs_eff, 2),
                    "comp_std=", float(np.std(component)),
                    "bpm_raw=", bpm_raw,
                    "quality=", quality
                )

            if bpm_raw is not None:
                self.last_bpm_raw = bpm_raw
                self._bpm_smooth.append(bpm_raw)
                self.current_bpm = round(float(np.mean(self._bpm_smooth)), 1)
                self.bpm_history.append(self.current_bpm)

        return self.get_status(quality_override=quality)

    def get_status(self, quality_override: Optional[float] = None) -> Dict:
        return {
            "face_detected": self.face_detected,
            "bpm": self.current_bpm,
            "bpm_raw": self.last_bpm_raw,
            "signal_quality": float(quality_override) if quality_override is not None else 0.0,
            "frames_processed": self.frame_idx,
            "fps_estimate": float(self._fps_est) if self._fps_est is not None else None,
            "roi_polygons": self.last_polygons,
            "ppg_waveform": list(self.green_wave)[-150:],   # keep payload small
            "bpm_history": list(self.bpm_history)[-150:],
        }