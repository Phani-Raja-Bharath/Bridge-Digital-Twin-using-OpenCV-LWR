# Bridge Camera Monitoring App - Setup Guide

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install streamlit opencv-python numpy pandas pillow
```

Or use requirements file:
```bash
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run bridge_camera_app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🎯 Features

### ✅ Automatic Video Playback
- No need to manually open VLC
- Live HLS stream plays directly in browser
- 5 Twin Bridges camera angles available

### 📸 Smart Snapshot Capture
- **Auto Mode**: Captures at user-defined intervals (5-300 seconds)
- **Manual Mode**: Click button to capture anytime
- Snapshots saved to `bridge_snapshots/` folder

### 📊 Real-Time Analytics
- Vehicle counting (basic, upgrade to YOLO)
- Traffic load estimation (lbs and tons)
- Running statistics (avg vehicles, avg load)
- Data table with all measurements

### 💾 Data Export
- Download CSV with all traffic data
- Includes timestamp, vehicle count, load estimates
- Ready for Monte Carlo simulation

---

## 🎬 How to Use

### Basic Usage

1. **Launch App:**
   ```bash
   streamlit run bridge_camera_app.py
   ```

2. **Select Camera:**
   - Choose from 5 Twin Bridges cameras in sidebar
   - Each shows different angle of the bridge

3. **Configure Snapshots:**
   - Set interval: 5-300 seconds
   - Enable "Auto Capture" for continuous monitoring

4. **Start Monitoring:**
   - Click "▶️ Start Stream"
   - Video plays automatically
   - Snapshots captured at intervals

5. **View Results:**
   - Live statistics on right panel
   - Data table below video
   - Latest measurement updates in real-time

6. **Export Data:**
   - Click "📥 Download CSV"
   - Use in your Monte Carlo simulation

### Advanced Usage

#### Collect 24-Hour Traffic Pattern
```bash
# Run app
streamlit run bridge_camera_app.py

# In app:
1. Select camera: "5821 - North of Mohawk"
2. Set interval: 60 seconds (1 snapshot/minute)
3. Enable auto-capture
4. Start stream
5. Let run for 24 hours
6. Download CSV (1,440 data points)
```

#### Multi-Camera Monitoring
Run multiple instances on different ports:

```bash
# Terminal 1 - Camera 5821
streamlit run bridge_camera_app.py --server.port 8501

# Terminal 2 - Camera 3645
streamlit run bridge_camera_app.py --server.port 8502

# Terminal 3 - Camera 3646
streamlit run bridge_camera_app.py --server.port 8503
```

---

## 📁 Output Files

### Snapshot Files
Location: `bridge_snapshots/`

Format: `{camera_number}_{timestamp}.jpg`

Example:
```
bridge_snapshots/
├── 5821_20250323_143052.jpg
├── 5821_20250323_143122.jpg
├── 5821_20250323_143152.jpg
└── ...
```

### Traffic Data CSV
Columns:
- `Timestamp`: Date and time of measurement
- `Camera`: Camera name/number
- `Vehicles`: Vehicle count
- `Load (lbs)`: Estimated load in pounds
- `Load (tons)`: Estimated load in tons
- `Snapshot`: Filename of associated image

Example:
```csv
Timestamp,Camera,Vehicles,Load (lbs),Load (tons),Snapshot
2025-03-23 14:30:52,5821 - North of Mohawk,12,48000,24.0,5821_20250323_143052.jpg
2025-03-23 14:31:22,5821 - North of Mohawk,15,60000,30.0,5821_20250323_143122.jpg
```

---

## 🔧 Customization

### Change Vehicle Detection
Replace `count_vehicles_simple()` function with YOLO:

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

def count_vehicles_yolo(frame):
    results = model(frame)
    vehicles = 0
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            # Count cars (2), trucks (7), buses (5)
            if class_id in [2, 5, 7]:
                vehicles += 1
    
    return vehicles
```

### Change Load Estimation
Modify `estimate_load()` function:

```python
def estimate_load(vehicle_count, truck_ratio=0.2):
    # More accurate: separate cars and trucks
    cars = int(vehicle_count * (1 - truck_ratio))
    trucks = int(vehicle_count * truck_ratio)
    
    car_weight = 4000  # lbs
    truck_weight = 35000  # lbs
    
    total_load = (cars * car_weight) + (trucks * truck_weight)
    return total_load / 2000  # tons
```

### Add New Cameras
Edit `TWIN_BRIDGES_CAMERAS` dictionary:

```python
TWIN_BRIDGES_CAMERAS = {
    "Your Camera Name": "https://stream-url.com/playlist.m3u8",
    # ... existing cameras
}
```

---

## 🌉 Integration with Bridge Project

### For Monte Carlo Simulation

```python
import pandas as pd
import numpy as np

# Load collected data
df = pd.read_csv('bridge_traffic_20250323_143000.csv')

# Extract traffic patterns
hourly_vehicles = df.groupby(df['Timestamp'].str[:13])['Vehicles'].mean()
hourly_load = df.groupby(df['Timestamp'].str[:13])['Load (tons)'].mean()

# Feed into Monte Carlo
traffic_distribution = df['Vehicles'].values
load_distribution = df['Load (tons)'].values

# Run simulation
simulated_loads = np.random.choice(load_distribution, size=10000)
fatigue_estimate = calculate_fatigue(simulated_loads)
```

### For Random Forest Model

```python
# Use as training features
features = df[['Vehicles', 'Load (tons)']]
target = calculate_maintenance_need(features)

# Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(features, target)
```

---

## 💡 Tips & Best Practices

### Data Collection
- **Morning Rush**: 6-9 AM for peak traffic
- **Evening Rush**: 4-7 PM for peak traffic
- **Off-Peak**: 10 PM - 5 AM for baseline
- **Weekday vs Weekend**: Collect both patterns

### Snapshot Intervals
- **Heavy Traffic**: 30 seconds (high resolution)
- **Normal Traffic**: 60 seconds (balanced)
- **Light Traffic**: 120 seconds (efficient)
- **24-Hour Pattern**: 300 seconds (overview)

### Performance
- Close app when not in use (stops video stream)
- Download data regularly (browser memory limit)
- Use single camera at a time for stability
- Clear old snapshots periodically

---

## 🐛 Troubleshooting

### Stream Won't Start
```
Problem: Black screen or "Stream not available"
Solution:
1. Check internet connection
2. Try different camera
3. Restart app
4. Check if NY511 is down: https://511ny.org
```

### High CPU Usage
```
Problem: Computer running slow
Solution:
1. Increase snapshot interval
2. Stop stream when not needed
3. Use single camera instance
4. Close other apps
```

### Snapshots Not Saving
```
Problem: No files in bridge_snapshots/
Solution:
1. Check if folder exists
2. Verify write permissions
3. Check disk space
4. Look for error messages in terminal
```

### Vehicle Count Inaccurate
```
Problem: Count seems wrong
Solution:
1. This is expected - basic detection only
2. Upgrade to YOLO for accuracy
3. Manually verify with snapshots
4. Adjust detection thresholds
```

---

## 📚 Next Steps

### Upgrade Vehicle Detection
1. Install YOLOv8: `pip install ultralytics`
2. Download model: `yolo download yolov8n.pt`
3. Replace detection function (see Customization section)

### Deploy to Cloud
```bash
# Deploy to Streamlit Cloud
1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect repository
4. Deploy app
```

### Add Machine Learning
```python
# Train model on collected data
from sklearn.ensemble import RandomForestRegressor

# Load your data
df = pd.read_csv('bridge_traffic_data.csv')

# Train fatigue prediction model
model = RandomForestRegressor()
model.fit(df[['Vehicles', 'Load (tons)']], df['Fatigue_Index'])
```

---

## 📞 Support

For issues or questions about:
- **App functionality**: Check troubleshooting section
- **NY511 API**: Visit https://511ny.org/developers
- **Bridge project integration**: Refer to Monte Carlo simulation docs
- **YOLO integration**: See https://docs.ultralytics.com

---

## 📄 License

This tool is for educational and research purposes.
Traffic camera streams provided by NY511.
