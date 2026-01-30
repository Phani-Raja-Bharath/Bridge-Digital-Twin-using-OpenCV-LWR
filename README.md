# Hybrid Digital Twin – Bridge Fatigue Monitoring (Computer Vision + Traffic Simulation)

This repository contains a proof-of-concept “hybrid digital twin” pipeline that estimates bridge loading and fatigue proxies using:
- Live traffic camera frames (computer vision vehicle detection with YOLOv8)
- Lightweight traffic modeling (density proxy + optional LWR-style simulation during monitoring sessions)
- Environmental stress proxies (weather + freeze–thaw, precipitation, etc.)
- Fatigue and reliability proxies (Miner’s-rule style damage accumulation and a reliability index β)
- Simple validation support against NYSDOT traffic count station data (manual entry via a generated HTML template)

There are two single-file applications included in this project (same core workflow, different bridge/camera sources).

## Files

- `TwinBridge_Traffic_cam.py`  
  Twin Bridges (I‑87 over Mohawk River, New York) implementation.

- `Peace_Bridge.py`  
  Peace Bridge (Buffalo, NY ↔ Fort Erie, ON) implementation.

## Key difference between the two files

### 1) Camera feed source and how the frame is obtained

- `TwinBridge_Traffic_cam.py` uses direct HLS camera streams (`playlist.m3u8`) from NYSDOT’s SkyVDN endpoints (example: `https://s51.nysdot.skyvdn.com/.../playlist.m3u8`).  
  This is already an HLS URL, so the app can directly pass it to `ffmpeg` to grab a frame.

- `Peace_Bridge.py` uses a YouTube live link (example: `https://youtu.be/...`) as its camera source.  
  The app first resolves the YouTube URL to a direct stream URL using `yt-dlp`, then uses `ffmpeg` to capture a frame.

## Workflow

1. Capture a single frame from the selected traffic camera
2. Run YOLOv8 detection on the frame to classify vehicles (car/truck/bus/motorcycle)
3. Split detections into “approaching” vs “past” based on a lane divider + camera orientation
4. Estimate a proxy bridge load (tons) using user-adjustable weights
5. Compute fatigue proxy scores (traffic stress + environmental stress + combined score)
6. Optional monitoring session: capture frames periodically, log results, run simulation and compute:
   - Fatigue damage (Miner proxy)
   - Reliability index β
7. Export results:
   - Monitoring session logs to CSV
   - HTML report(s) / templates for validation and documentation

## Requirements

### Software
- Python 3.9+
- `ffmpeg` installed and available on PATH  
  - Windows: install via Chocolatey (`choco install ffmpeg`) or download binaries and add to PATH
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt-get install ffmpeg`

### APIs / Data Sources
- YOLOv8 (Ultralytics) for vehicle detection
- Open-Meteo (current + historical weather)
- YouTube Live (camera source used by `Peace_Bridge.py`)
- HLS `.m3u8` streams (camera source used by `TwinBridge_Traffic_cam.py`)

### Tools
- `yt-dlp` (required only for YouTube camera source in `Peace_Bridge.py`)
- `ffmpeg` (frame capture for both apps)

### Python packages
Minimum (typical):
- streamlit
- opencv-python
- numpy
- pandas
- scipy
- requests
- pillow
- ultralytics

Optional features:
- plotly (for interactive charts)
- scikit-learn (for Random Forest components used inside the apps)

If you plan to run `Peace_Bridge.py` (YouTube camera source: PeaceBridgeAuthority), install:
- yt-dlp

## Installation

Create and activate a virtual environment, then install dependencies.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install --upgrade pip
pip install streamlit opencv-python numpy pandas scipy requests pillow ultralytics
pip install plotly scikit-learn  # optional
pip install yt-dlp               # required for Peace_Bridge.py (Credits YouTube: PeaceBridgeAuthority)
```

Verify `ffmpeg`:
```bash
ffmpeg -version
```

## How to run

Run either file using Streamlit.

### Twin Bridges (HLS camera)
```bash
streamlit run TwinBridge_Traffic_cam.py
```

### Peace Bridge (YouTube camera)
```bash
streamlit run Peace_Bridge.py
```

## Inputs (UI controls)

Common controls (both apps):
- Camera selector (if multiple configured)
- Lane Divider (splits left/right lane regions for approaching/past labeling)
- Detection confidence threshold
- ROI box enable + ROI boundary sliders (counts vehicles only in the bridge-deck region)
- Vehicle weights (car/truck/bus/motorcycle) used to compute load proxy

Monitoring session controls:
- Capture interval (seconds/minutes)
- Session duration (minutes)
- Simulation parameters (e.g., Monte Carlo runs, jam probability where available)

## Outputs

On-screen:
- Live frame (annotated with detections)
- Vehicle counts (cars/trucks/buses/motorcycles) and totals
- Estimated load proxy (tons)
- Weather and environmental stress indicators
- Fatigue proxy metrics and (when session is running) reliability index β / damage

Download/export:
- CSV of monitoring session log (vehicle totals, loads, damage, reliability index, timestamps)
- HTML report(s) (summary + validation template for comparing with NYSDOT counts)
- Optional summary text export (if enabled in the app)

## Notes and limitations

- This is a proof-of-concept for research and demonstration.
- Camera-based “load” and fatigue are proxy metrics (not calibrated to weigh‑in‑motion or structural sensor data).
- Accuracy depends heavily on camera angle, visibility, ROI selection, and YOLO model performance.
- If the camera feed is unavailable, the apps may fall back to demo-mode values.

## Troubleshooting

- If you see “Camera unavailable”:
  - Confirm the camera URL is still valid.
  - Confirm `ffmpeg` is installed and on PATH.
  - For `Peace_Bridge.py`, confirm `yt-dlp` is installed and accessible.

- If YOLO errors:
  - Ensure `pip install ultralytics` worked.
  - The first run may download model weights.

## Citation / attribution

If you use this code in academic work, cite your project/repository and note:
- YOLOv8 via Ultralytics
- Weather data source used in the code (see `fetch_weather` implementation inside the file)
