import base64
import os
import uuid
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from heart_rate_rppg import RPPGEngine

app = FastAPI(title="rPPG Liveness API", version="1.0.0")

# Configure model path via env or default
DEFAULT_MODEL_PATH = os.getenv("FACE_LANDMARKER_MODEL", "models/face_landmarker.task")

SESSIONS: Dict[str, RPPGEngine] = {}

class CreateSessionResponse(BaseModel):
    session_id: str

class FrameIn(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded JPEG/PNG image bytes")

class StatusOut(BaseModel):
    face_detected: bool
    bpm: float
    bpm_raw: Optional[float] = None
    signal_quality: float
    frames_processed: int
    fps_estimate: Optional[float] = None
    roi_polygons: Dict
    ppg_waveform: list
    bpm_history: list

@app.get("/")
def root():
    return {"message": "rPPG Liveness API is running. Go to /docs", "model_path": DEFAULT_MODEL_PATH}

@app.get("/health")
def health():
    return {"ok": True, "sessions": len(SESSIONS), "model_path": DEFAULT_MODEL_PATH}

@app.post("/session", response_model=CreateSessionResponse)
def create_session():
    sid = str(uuid.uuid4())
    try:
        SESSIONS[sid] = RPPGEngine(
                        model_path=DEFAULT_MODEL_PATH,
                        fs=10,  
                        buffer_seconds=20,
                        band_low=0.7,             
                        band_high=3.0,       
                        update_every_frames=5
                    )
    except Exception as e:
        # Surface the real error (missing model, mediapipe tasks, etc.)
        raise HTTPException(status_code=500, detail=f"Failed to create session: {e}")
    return {"session_id": sid}

@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    if session_id in SESSIONS:
        del SESSIONS[session_id]
        return {"deleted": True}
    return {"deleted": False}

@app.get("/session/{session_id}/status", response_model=StatusOut)
def get_status(session_id: str):
    engine = SESSIONS.get(session_id)
    if not engine:
        raise HTTPException(status_code=404, detail="Session not found")
    return engine.get_status()

@app.post("/session/{session_id}/frame", response_model=StatusOut)
def push_frame(session_id: str, frame_in: FrameIn):
    engine = SESSIONS.get(session_id)
    if not engine:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        img_bytes = base64.b64decode(frame_in.image_b64)
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image_b64: {e}")

    try:
        return engine.process_frame(frame)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Frame processing failed: {e}")