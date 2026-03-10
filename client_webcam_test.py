import base64
import time
import cv2
import requests
import numpy as np

API = "http://127.0.0.1:8000"   
SEND_FPS = 10                  
JPEG_QUALITY = 80
DRAW_ROIS = True                


def create_session() -> str:
    try:
        r = requests.post(f"{API}/session", timeout=10)
        r.raise_for_status()
        return r.json()["session_id"]
    except requests.exceptions.RequestException as e:
        raise SystemExit(
            f"\nCannot create session at {API}/session\n"
            f"Make sure the server is running:\n"
            f"  uvicorn api:app --host 127.0.0.1 --port 8000 --reload\n"
            f"And check:\n"
            f"  curl {API}/health\n\n"
            f"Original error: {e}\n"
        )


def delete_session(session_id: str) -> None:
    try:
        requests.delete(f"{API}/session/{session_id}", timeout=5)
    except Exception:
        pass


def encode_frame_to_b64(frame_bgr) -> str:
    ok, jpg = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    if not ok:
        raise ValueError("Failed to encode frame to JPEG")
    return base64.b64encode(jpg.tobytes()).decode("utf-8")


def send_frame(session_id: str, frame_bgr):
    b64 = encode_frame_to_b64(frame_bgr)
    r = requests.post(
        f"{API}/session/{session_id}/frame",
        json={"image_b64": b64},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()


def draw_roi_polygons(frame_bgr, roi_polygons: dict):
    """
    roi_polygons format:
      {
        "forehead": [[x,y], ...],
        "left_cheek": [[x,y], ...],
        "right_cheek": [[x,y], ...]
      }
    Colors:
      forehead = green, left_cheek = blue, right_cheek = red
    """
    COLORS = {
        "forehead": (0, 255, 0),
        "left_cheek": (255, 0, 0),
        "right_cheek": (0, 0, 255),
    }
    for name, pts in (roi_polygons or {}).items():
        if not pts or len(pts) < 3:
            continue
        pts_np = cv2.UMat(np.array(pts, dtype="int32")).get()
        cv2.polylines(frame_bgr, [pts_np], True, COLORS.get(name, (0, 255, 255)), 2)
        # label
        x, y = pts_np[0]
        cv2.putText(frame_bgr, name, (int(x), int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS.get(name, (0, 255, 255)), 2)


def draw_waveform_overlay(frame, samples, x=20, y=200, w=420, h=120, color=(0,255,0), label="PPG"):
    """Draw a simple scrolling waveform box."""
    if not samples or len(samples) < 5:
        return

    box = frame.copy()
    cv2.rectangle(box, (x, y), (x+w, y+h), (25, 25, 25), -1)
    cv2.addWeighted(box, 0.6, frame, 0.4, 0, frame)

    arr = np.array(samples, dtype=np.float32)
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn < 1e-6:
        cv2.putText(frame, f"{label} (flat)", (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return

    norm = (arr - mn) / (mx - mn)
    step = w / max(1, (len(norm)-1))
    pts = []
    for i, v in enumerate(norm):
        px = int(x + i * step)
        py = int(y + h - v * h)
        pts.append((px, py))

    for i in range(1, len(pts)):
        cv2.line(frame, pts[i-1], pts[i], color, 2)

    cv2.putText(frame, label, (x, y-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_bpm_trend_overlay(frame, bpm_hist, x=20, y=340, w=420, h=120, color=(0,0,255), label="BPM"):
    """Draw BPM trend box."""
    if not bpm_hist or len(bpm_hist) < 3:
        return

    box = frame.copy()
    cv2.rectangle(box, (x, y), (x+w, y+h), (25, 25, 25), -1)
    cv2.addWeighted(box, 0.6, frame, 0.4, 0, frame)

    arr = np.array(bpm_hist, dtype=np.float32)
    
    arr = np.clip(arr, 40, 180)
    mn, mx = 40.0, 180.0
    norm = (arr - mn) / (mx - mn + 1e-6)

    step = w / max(1, (len(norm)-1))
    pts = []
    for i, v in enumerate(norm):
        px = int(x + i * step)
        py = int(y + h - v * h)
        pts.append((px, py))

    for i in range(1, len(pts)):
        cv2.line(frame, pts[i-1], pts[i], color, 2)

    cv2.putText(frame, label, (x, y-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    

def main():
    session_id = create_session()
    print("Session created:", session_id)
    print("Streaming frames... (press 'q' to quit)\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        delete_session(session_id)
        raise SystemExit("Could not open webcam (index 0). Try another index.")

    send_interval = 1.0 / float(SEND_FPS)
    last_send = 0.0

    # simple stats
    last_print = 0.0
    print_interval = 0.5

    latest = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()

            # Send at ~SEND_FPS
            if now - last_send >= send_interval:
                last_send = now
                try:
                    latest = send_frame(session_id, frame)
                except requests.exceptions.RequestException as e:
                    print("Request error:", e)
                except Exception as e:
                    print("Client error:", e)

            # Overlay debug info + ROIs + waveforms
            if latest is not None:
                import numpy as np  

                bpm = latest.get("bpm", 0.0)
                quality = latest.get("signal_quality", 0.0)
                face = latest.get("face_detected", False)
                fps_est = latest.get("fps_estimate", None)

                # Top-left HUD
                cv2.putText(frame, f"BPM: {bpm}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.putText(frame, f"Quality: {quality:.2f}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Face: {face}", (20, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                if fps_est is not None:
                    cv2.putText(frame, f"API FPS est: {fps_est:.1f}", (20, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Draw ROIs returned by API
                if DRAW_ROIS:
                    roi_polys = latest.get("roi_polygons", {})
                    for name, pts in (roi_polys or {}).items():
                        if not pts or len(pts) < 3:
                            continue
                        pts_np = np.array(pts, dtype=np.int32)

                        if name == "forehead":
                            color = (0, 255, 0)     # green
                        elif name == "left_cheek":
                            color = (255, 0, 0)     # blue
                        elif name == "right_cheek":
                            color = (0, 0, 255)     # red
                        else:
                            color = (0, 255, 255)   # yellow

                        cv2.polylines(frame, [pts_np], True, color, 2)
                        x, y = pts_np[0]
                        cv2.putText(frame, name, (int(x), int(y) - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Draw waveform + BPM trend overlays at bottom
                ppg = latest.get("ppg_waveform", [])
                bpm_hist = latest.get("bpm_history", [])

                draw_waveform_overlay(frame, ppg, x=20, y=200,
                                      w=420, h=120, color=(0, 255, 0),
                                      label="PPG (Green)")
                draw_bpm_trend_overlay(frame, bpm_hist, x=20, y=340,
                                       w=420, h=120, color=(0, 0, 255),
                                       label="BPM (Red)")

                
                if now - last_print >= print_interval:
                    last_print = now
                    print(
                        f"BPM={bpm} | quality={quality:.2f} | face={face} | "
                        f"frames={latest.get('frames_processed')} | "
                        f"ppg_len={len(ppg)} | bpm_hist_len={len(bpm_hist)}"
                    )

            cv2.imshow("Client Webcam → rPPG API (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        delete_session(session_id)
        print("\n Session cleaned up. Bye.")


if __name__ == "__main__":
    main()