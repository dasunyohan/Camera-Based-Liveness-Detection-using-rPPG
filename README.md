Camera-Based Liveness Detection using rPPG
==========================================

This project implements a **real-time liveness detection system** for identity verification using **remote Photoplethysmography (rPPG)**.It verifies live human presence using **only a standard RGB camera**, without wearables or additional hardware.

The system extracts subtle physiological signals from facial videos and uses them as a strong liveness cue against presentation and replay attacks.

📌 Overview
-----------

Traditional liveness checks (blink detection, face movement) are increasingly vulnerable to **replay attacks and AI-generated faces**.This project takes a **physiology-aware approach**, leveraging real blood-volume pulse signals captured from facial skin regions.

**Key idea:**A real human must obey **biological, physical, and temporal constraints** that are difficult to fake in real time.

🧠 How It Works (Pipeline)
--------------------------

1.  **Face & ROI Detection**
    
    *   Detects the face using MediaPipe Face Mesh
        
    *   Tracks stable skin regions: **forehead + both cheeks**
        
2.  **RGB Time-Series Extraction**
    
    *   Mean RGB values are extracted from each ROI per frame
        
    *   Produces synchronized multivariate time-series signals
        
3.  **Remote Photoplethysmography (rPPG)**
    
    *   Subtle color variations caused by blood flow are captured
        
    *   Enables non-contact physiological measurement
        
4.  **Signal Separation (ICA / PCA)**
    
    *   RGB signals are mixtures of pulse, motion, lighting, and noise
        
    *   Blind source separation isolates the dominant pulse component
        
5.  **Signal Processing & BPM Estimation**
    
    *   Band-pass filtering (0.7–4 Hz) isolates heart-rate frequencies
        
    *   Frequency analysis estimates **beats per minute (BPM)**
        
6.  **Multi-ROI Consistency Check**
    
    *   Pulse consistency across forehead and cheeks
        
    *   Strengthens liveness confidence
        

🎥 Demo Visualization
---------------------

In the demo video:

*   **Green waveform** → Extracted **PPG (Photoplethysmography) signal**
    
*   **Red waveform** → **Smoothed average heart rate (BPM)** over time
    

These visualizations demonstrate live physiological signal extraction from the face.

🔐 Application to Liveness Detection
------------------------------------

This system is designed to resist:

*   Printed photo attacks
    
*   Screen-based replay attacks
    
*   Many pre-recorded and synthetic face attacks
    

By focusing on **physiological realism**, the system raises the difficulty of spoofing without increasing user friction.

Note: No biometric system is perfect. This project demonstrates a **multi-signal liveness approach**, not absolute AI detection.

🛠️ Tech Stack
--------------

*   **Python**
    
*   **OpenCV**
    
*   **MediaPipe Face Mesh**
    
*   **NumPy / SciPy**
    
*   **scikit-learn (ICA / PCA)**
    
*   **Signal Processing (FFT, Bandpass Filtering)**
    

📂 Project Structure
--------------------

` ├── realtime_rppg_multi_roi.py # Main real-time liveness detection script `<br>`
├── README.md # Project documentation  `

🚀 Getting Started
------------------

### 1️⃣ Install Dependencies

`   pip install opencv-python mediapipe numpy scipy scikit-learn   `

### 2️⃣ Run the System

`   python realtime_rppg_multi_roi.py   `

### 3️⃣ Usage

*   Ensure good frontal lighting
    
*   Keep face visible for ~8–12 seconds
    
*   Press q to exit
    

⚠️ Limitations
--------------

*   Performance depends on lighting conditions
    
*   Large head motion may degrade signal quality
    
*   Not designed to defeat high-budget real-time face reenactment systems
