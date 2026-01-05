import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt
import time

# Optional imports
try:
    from sklearn.decomposition import FastICA, PCA
    SKLEARN_AVAILABLE = True
except:
    SKLEARN_AVAILABLE = False


FS = 30
BUFFER_SECONDS = 12
BUFFER_SIZE = FS * BUFFER_SECONDS

BANDPASS_LOW = 0.7
BANDPASS_HIGH = 4.0

UPDATE_EVERY_FRAMES = 10
BPM_SMOOTH_WINDOW = 5

GRAPH_W = 420
GRAPH_H = 120
GRAPH_LEN = 300

# ROI definitions
FOREHEAD = [10, 338, 297, 332]
LEFT_CHEEK = [234, 93, 132, 58]
RIGHT_CHEEK = [454, 323, 361, 288]

ROI_DEFS = [
    ("forehead", FOREHEAD, (0,255,0)),     # green
    ("left_cheek", LEFT_CHEEK, (255,0,0)), # blue
    ("right_cheek", RIGHT_CHEEK, (0,0,255))# red
]


def bandpass_filter(sig):
    nyq = FS / 2
    b, a = butter(3, [BANDPASS_LOW/nyq, BANDPASS_HIGH/nyq], btype='band')
    return filtfilt(b, a, sig)

def estimate_bpm(signal):
    if len(signal) < FS * 5:
        return None
    sig = bandpass_filter(signal - np.mean(signal))
    freqs = np.fft.rfftfreq(len(sig), 1/FS)
    fft_vals = np.abs(np.fft.rfft(sig))
    mask = (freqs >= BANDPASS_LOW) & (freqs <= BANDPASS_HIGH)
    if not np.any(mask):
        return None
    return freqs[mask][np.argmax(fft_vals[mask])] * 60

# ROI Extraction
def extract_multi_roi_rgb(frame, landmarks):
    h, w, _ = frame.shape
    roi_rgbs = []
    polygons = []

    for _, indices, _ in ROI_DEFS:
        pts = np.array([
            (int(landmarks[i].x * w), int(landmarks[i].y * h))
            for i in indices
        ], dtype=np.int32)

        mask = np.zeros((h,w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        pixels = frame[mask == 255]

        if len(pixels) == 0:
            continue

        mean_bgr = np.mean(pixels, axis=0)
        roi_rgbs.append(mean_bgr[::-1])  # RGB
        polygons.append(pts)

    if not roi_rgbs:
        return None, polygons

    fused_rgb = np.mean(roi_rgbs, axis=0)
    return fused_rgb, polygons


mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1)
cap = cv2.VideoCapture(0)

rgb_buffer = deque(maxlen=BUFFER_SIZE)
green_wave = deque(maxlen=GRAPH_LEN)
bpm_buffer = deque(maxlen=BPM_SMOOTH_WINDOW)
bpm_history = deque(maxlen=GRAPH_LEN)

frame_idx = 0
current_bpm = 0.0
start_t = time.time()

print("Starting multi-ROI rPPG. Press 'q' to quit.")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    vis = frame.copy()

    fused_rgb = None
    polygons = []

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        fused_rgb, polygons = extract_multi_roi_rgb(frame, landmarks)

        for (name, _, color), pts in zip(ROI_DEFS, polygons):
            cv2.polylines(vis, [pts], True, color, 2)

    if fused_rgb is not None:
        rgb_buffer.append(fused_rgb)
        green_wave.append(fused_rgb[1])
    elif rgb_buffer:
        rgb_buffer.append(rgb_buffer[-1])
        green_wave.append(green_wave[-1])

 
    if frame_idx % UPDATE_EVERY_FRAMES == 0 and len(rgb_buffer) >= FS * 5:
        data = np.array(rgb_buffer)
        data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)

        component = data[:,1]  # fallback green

        if SKLEARN_AVAILABLE:
            try:
                ica = FastICA(n_components=3, max_iter=500)
                comps = ica.fit_transform(data)
                powers = []
                for i in range(3):
                    filt = bandpass_filter(comps[:,i])
                    powers.append(np.std(filt))
                component = comps[:, np.argmax(powers)]
            except:
                pass

        bpm = estimate_bpm(component)
        if bpm:
            bpm_buffer.append(bpm)
            current_bpm = round(np.mean(bpm_buffer), 1)
            bpm_history.append(current_bpm)


    cv2.putText(vis, f"BPM: {current_bpm}", (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

    def draw_graph(data, x, y, w, h, color):
        if len(data) < 2:
            return
        arr = np.array(data)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        step = w / (len(arr)-1)
        for i in range(1, len(arr)):
            cv2.line(vis,
                     (int(x+(i-1)*step), int(y+h-arr[i-1]*h)),
                     (int(x+i*step), int(y+h-arr[i]*h)),
                     color, 2)

    draw_graph(green_wave, 30, vis.shape[0]-260, GRAPH_W, GRAPH_H, (0,255,0))
    draw_graph(bpm_history, 30, vis.shape[0]-130, GRAPH_W, GRAPH_H, (0,0,255))

    cv2.imshow("Multi-ROI rPPG", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
