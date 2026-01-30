import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import time
import subprocess
from PIL import Image
import io
import logging
import requests
from scipy import stats
import base64
import os   
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from ultralytics import YOLO  # type: ignore[attr-defined]
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# 1) APP CONFIGURATION
# =============================================================================

@dataclass
class BridgeConfig:
    """Peace Bridge specifications"""
    name: str = "Peace Bridge - US-Canada Border"
    location: str = "Buffalo, NY to Fort Erie, ON"
    latitude: float = 42.9069
    longitude: float = -78.9053
    year_built: int = 1927
    main_span_m: float = 176.8  # 580 ft main span
    total_length_m: float = 1768.0  # 5,800 ft total length
    material: str = "Steel arch bridge"
    daily_traffic: int = 8000  # Commercial vehicles per day (approximate)
    
    @property
    def age_years(self) -> int:
        return datetime.now().year - self.year_built


# Camera configurations
CAMERAS = {
    "Peace Bridge - Canada Bound": {
        "url": "https://youtu.be/DnUFAShZKus",
        "approaching_side": "right",
        "left_label": "USA Bound (PAST)",
        "right_label": "Canada Bound (LOAD)"
    }
}

VEHICLE_WEIGHTS = {'car': 4000, 'truck': 35000, 'bus': 25000, 'motorcycle': 500}
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

# =============================================================================
# 2) ROI CONFIGURATION
# =============================================================================

# Default ROI box as percentage of frame [x1%, y1%, x2%, y2%]
# This defines the "bridge deck" area where vehicles are counted
DEFAULT_ROI = {
    "Peace Bridge - Canada Bound": {
        "x1_pct": 0.20,  # Left edge (20% from left)
        "y1_pct": 0.35,  # Top edge (35% from top)
        "x2_pct": 0.80,  # Right edge (80% from left)
        "y2_pct": 0.75,  # Bottom edge (75% from top)
    }
}

# Camera calibration for distance estimation
# Based on typical DOT camera specs and lane widths
CAMERA_CALIBRATION = {
    "focal_length_px": 800,       # Estimated focal length in pixels
    "typical_car_width_m": 1.8,   # Average car width
    "typical_truck_width_m": 2.5, # Average truck width
    "typical_motorcycle_width_m": 0.8, # Average motorcycle width
    "lane_width_m": 3.6,          # Standard lane width
    "camera_height_m": 10,        # Estimated camera mounting height
    "camera_fov_deg": 60,         # Approximate field of view
}


def estimate_vehicle_distance(
    bbox_width_px: int,
    frame_width_px: int,
    vehicle_type: str = "car",
    focal_length_px: Optional[float] = None,
) -> Dict:
    """
    Estimate distance to vehicle based on bounding box size.
    
    Uses pinhole camera model:
    distance = (real_width × focal_length) / bbox_width
    
    Returns dict with distance estimate and confidence.
    """
    
    # Get real-world width based on vehicle type
    if vehicle_type == "truck":
        real_width = CAMERA_CALIBRATION["typical_truck_width_m"]
    elif vehicle_type == "bus":
        real_width = 2.5
    elif vehicle_type == "motorcycle":
        real_width = CAMERA_CALIBRATION["typical_motorcycle_width_m"]
    else:
        real_width = CAMERA_CALIBRATION["typical_car_width_m"]
    
    # Estimate focal length from FOV if not calibrated
    if focal_length_px is not None:
        focal_length = float(focal_length_px)
    else:
        fov_rad = np.radians(CAMERA_CALIBRATION["camera_fov_deg"])
        focal_length = (frame_width_px / 2) / np.tan(fov_rad / 2)
    
    # Calculate distance (clamp bbox width to reduce blow-ups)
    min_width_px = 10
    safe_width = max(int(bbox_width_px), min_width_px)
    distance_m = (real_width * focal_length) / safe_width
    
    # Estimate angle from center
    # (This would need bbox center_x, adding as placeholder)
    
    # Confidence based on bbox size (larger = more confident)
    confidence = min(1.0, bbox_width_px / 100)
    
    return {
        "distance_m": round(distance_m, 1),
        "confidence": round(confidence, 2),
        "method": "pinhole_model"
    }


def get_roi_pixels(frame_shape: Tuple, roi_pct: Dict) -> Tuple[int, int, int, int]:
    """Convert ROI percentages to pixel coordinates"""
    height, width = frame_shape[:2]
    x1 = int(width * roi_pct["x1_pct"])
    y1 = int(height * roi_pct["y1_pct"])
    x2 = int(width * roi_pct["x2_pct"])
    y2 = int(height * roi_pct["y2_pct"])
    return x1, y1, x2, y2


def is_in_roi(center_x: int, center_y: int, roi: Tuple[int, int, int, int]) -> bool:
    """Check if point is inside ROI box"""
    x1, y1, x2, y2 = roi
    return x1 <= center_x <= x2 and y1 <= center_y <= y2

# =============================================================================
# 3) WEATHER API (OPEN-METEO)
# =============================================================================

def fetch_weather(lat: float, lon: float) -> Dict:
    """Fetch current weather from Open-Meteo API"""
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
            f"&past_days=7&forecast_days=1"
            f"&timezone=America/New_York"
        )
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        current = data.get("current", {})
        daily = data.get("daily", {})
        
        # Count freeze-thaw cycles (temp crossing 0°C)
        freeze_thaw = 0
        if daily.get("temperature_2m_min") and daily.get("temperature_2m_max"):
            for tmin, tmax in zip(daily["temperature_2m_min"], daily["temperature_2m_max"]):
                if tmin is not None and tmax is not None:
                    if tmin < 0 < tmax:
                        freeze_thaw += 1
        
        return {
            "temperature": current.get("temperature_2m", 0),
            "humidity": current.get("relative_humidity_2m", 50),
            "precipitation": current.get("precipitation", 0),
            "wind_speed": current.get("wind_speed_10m", 0),
            "freeze_thaw_7day": freeze_thaw,
            "success": True
        }
    except Exception as e:
        logger.error(f"Weather fetch failed: {e}")
        return {
            "temperature": 10,
            "humidity": 50,
            "precipitation": 0,
            "wind_speed": 5,
            "freeze_thaw_7day": 0,
            "success": False
        }


def fetch_historical_weather(lat: float, lon: float, months: int = 12) -> Optional[pd.DataFrame]:
    """
    Fetch historical weather from Open-Meteo Archive API.
    Free, no API key required.
    
    Returns daily data for the specified number of months.
    """
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=months * 30)
        
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
            f"rain_sum,snowfall_sum,wind_speed_10m_max"
            f"&timezone=America/New_York"
        )
        
        response = requests.get(url, timeout=30)
        data = response.json()
        
        daily = data.get("daily", {})
        
        if not daily.get("time"):
            return None
        
        df = pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            "temp_max": daily.get("temperature_2m_max", []),
            "temp_min": daily.get("temperature_2m_min", []),
            "precipitation": daily.get("precipitation_sum", []),
            "rain": daily.get("rain_sum", []),
            "snowfall": daily.get("snowfall_sum", []),
            "wind_max": daily.get("wind_speed_10m_max", [])
        })
        
        # Calculate freeze-thaw cycles
        df["freeze_thaw"] = ((df["temp_min"] < 0) & (df["temp_max"] > 0)).astype(int)
        
        # Calculate monthly aggregates
        df["month"] = df["date"].dt.to_period("M")  # type: ignore[attr-defined]
        df["year_month"] = df["date"].dt.strftime("%Y-%m")  # type: ignore[attr-defined]
        
        # Winter salt exposure (precipitation when temp < 5°C)
        df["salt_exposure"] = df.apply(
            lambda row: row["precipitation"] if row["temp_max"] < 5 else 0, axis=1
        )
        
        return df
        
    except Exception as e:
        logger.error(f"Historical weather fetch failed: {e}")
        return None


def analyze_historical_weather(df: pd.DataFrame) -> Dict:
    """Analyze historical weather for fatigue patterns"""
    
    if df is None or len(df) == 0:
        return {}
    
    # Monthly aggregates
    monthly = df.groupby("year_month").agg({
        "freeze_thaw": "sum",
        "precipitation": "sum",
        "salt_exposure": "sum",
        "temp_min": "min",
        "temp_max": "max",
        "wind_max": "max"
    }).reset_index()
    
    # Annual totals
    total_freeze_thaw = df["freeze_thaw"].sum()
    total_precipitation = df["precipitation"].sum()
    total_salt_exposure = df["salt_exposure"].sum()
    
    # Identify worst months
    worst_freeze_thaw_month = monthly.loc[monthly["freeze_thaw"].idxmax(), "year_month"] if len(monthly) > 0 else "N/A"
    worst_precip_month = monthly.loc[monthly["precipitation"].idxmax(), "year_month"] if len(monthly) > 0 else "N/A"
    
    # Seasonal analysis
    df["season"] = df["date"].dt.month.map({  # type: ignore[attr-defined]
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall"
    })
    
    seasonal = df.groupby("season").agg({
        "freeze_thaw": "sum",
        "salt_exposure": "sum",
        "precipitation": "sum"
    })
    
    # Winter contribution to corrosion
    winter_contribution = 0
    if total_salt_exposure > 0 and "Winter" in seasonal.index:
        winter_contribution = seasonal.loc["Winter", "salt_exposure"] / total_salt_exposure * 100
    
    return {
        "total_freeze_thaw": int(total_freeze_thaw),
        "total_precipitation_mm": round(total_precipitation, 1),
        "total_salt_exposure_mm": round(total_salt_exposure, 1),
        "worst_freeze_thaw_month": worst_freeze_thaw_month,
        "worst_precip_month": worst_precip_month,
        "winter_corrosion_contribution": round(winter_contribution, 1),
        "monthly_data": monthly,
        "seasonal_data": seasonal
    }

def generate_training_data_from_historical(
    historical_df: pd.DataFrame,
    road_length_m: float = 237.4,
    scenarios_per_day: int = 5
) -> pd.DataFrame:
    """
    Generate ML training data by combining historical weather with traffic scenarios.
    
    This is the RIGHT approach:
    - Historical weather provides REAL environmental conditions
    - Traffic scenarios simulate various load conditions
    - Combined fatigue is calculated for each combination
    - RF learns the relationship between (weather + traffic) → fatigue
    
    Returns DataFrame with features and target for ML training.
    """
    
    if historical_df is None or len(historical_df) == 0:
        logger.warning("No historical data for training")
        return pd.DataFrame()
    
    training_records = []
    
    for _, weather_row in historical_df.iterrows():
        # Extract weather features for this day
        temp = weather_row.get("temp_max", 15)
        temp_min = weather_row.get("temp_min", 10)
        precip = weather_row.get("precipitation", 0) or 0
        wind = weather_row.get("wind_max", 10) or 10
        freeze_thaw = weather_row.get("freeze_thaw", 0)
        snowfall = weather_row.get("snowfall", 0) or 0
        
        # Derived features
        date = weather_row.get("date", datetime.now())
        month = date.month if hasattr(date, 'month') else 6
        is_winter = 1 if month in [11, 12, 1, 2, 3] else 0
        temp_range = abs(temp - temp_min)
        
        # Salt exposure: precipitation when cold
        salt_exposure = precip * 3.0 if (is_winter and temp < 5) else precip * 0.5
        
        # Generate multiple traffic scenarios for this weather day
        for _ in range(scenarios_per_day):
            # Random traffic conditions
            density = np.random.uniform(0.02, 0.15)  # vehicles/meter
            v_max = np.random.choice([60, 80, 100, 120]) / 3.6  # km/h to m/s
            truck_pct = np.random.uniform(0.10, 0.25)
            
            # Run LWR simulation
            sim_result = run_lwr_simulation(
                initial_density=density,
                road_length_m=road_length_m,
                v_max_mps=v_max,
                inject_jam=np.random.random() < 0.3
            )
            
            # Calculate traffic stress (from simulation)
            traffic_stress = sim_result["fatigue"]
            shockwave = sim_result["shockwave_speed"]
            
            # Calculate environmental stress (weather-based)
            # Using actual formulas, not hardcoded
            freeze_thaw_stress = min(100, freeze_thaw * 30)  # Each F/T cycle adds stress
            humidity_stress = 50  # Assumed average, could enhance with humidity data
            temp_stress = 20 + abs(temp - 15) * 2 + temp_range * 1.5
            temp_stress = min(100, temp_stress)
            precip_stress = min(100, salt_exposure * 5)
            wind_stress = min(100, wind * 2)
            
            # Combined environmental (weighted)
            env_stress = (
                freeze_thaw_stress * 0.30 +
                humidity_stress * 0.15 +
                temp_stress * 0.20 +
                precip_stress * 0.25 +
                wind_stress * 0.10
            )
            
            # Age factor (Twin Bridges = 66 years)
            age_factor = 1.16  # 1.0 + (66-50)*0.01
            env_stress *= age_factor
            env_stress = min(100, env_stress)
            
            # Combined fatigue target (what RF will learn to predict)
            combined_fatigue = traffic_stress * 0.7 + env_stress * 0.3
            combined_fatigue = min(100, combined_fatigue)
            
            # Store training record
            training_records.append({
                # Traffic features
                "density": round(density, 4),
                "v_max": round(v_max * 3.6, 1),  # Back to km/h
                "truck_pct": round(truck_pct, 2),
                "shockwave_speed": round(shockwave, 4),
                
                # Weather features (REAL historical data)
                "temperature": round(temp, 1),
                "temp_min": round(temp_min, 1),
                "temp_range": round(temp_range, 1),
                "precipitation": round(precip, 1),
                "snowfall": round(snowfall, 1),
                "wind_speed": round(wind, 1),
                "freeze_thaw": freeze_thaw,
                "salt_exposure": round(salt_exposure, 1),
                "month": month,
                "is_winter": is_winter,
                
                # Intermediate calculations (for debugging)
                "traffic_stress": round(traffic_stress, 2),
                "env_stress": round(env_stress, 2),
                
                # TARGET
                "fatigue": round(combined_fatigue, 2)
            })
    
    df = pd.DataFrame(training_records)
    logger.info(f"Generated {len(df)} training samples from {len(historical_df)} days × {scenarios_per_day} scenarios")
    
    return df

def calculate_environmental_stress(
    temperature: float,
    humidity: float,
    precipitation: float,
    freeze_thaw_cycles: int,
    wind_speed: float,
    bridge_age: int
) -> Dict:
    """
    Calculate environmental stress factors for steel bridge corrosion.
    Based on ISO 9223 corrosivity categories and engineering literature.
    
    Returns normalized scores (0-100) for each factor and combined score.
    """
    
    # 1. Freeze-thaw stress (0-100)
    # 50+ cycles/year is severe for NY climate
    freeze_thaw_annual = freeze_thaw_cycles * 52 / 7  # Extrapolate to annual
    freeze_thaw_score = min(100, freeze_thaw_annual * 1.5)
    
    # 2. Humidity corrosion factor (0-100)
    # Steel corrosion accelerates above 60% RH
    if humidity < 60:
        humidity_score = humidity * 0.5
    else:
        humidity_score = 30 + (humidity - 60) * 1.75
    humidity_score = min(100, humidity_score)
    
    # 3. Temperature stress (0-100)
    # Thermal cycling and extreme temps
    if temperature < -10:
        temp_score = 80 + abs(temperature + 10) * 2
    elif temperature < 0:
        temp_score = 50 + abs(temperature) * 3
    elif temperature > 35:
        temp_score = 50 + (temperature - 35) * 3
    else:
        temp_score = 20 + abs(temperature - 15) * 1.5
    temp_score = min(100, temp_score)
    
    # 4. Precipitation/salt factor (0-100)
    # Winter precipitation = road salt = chloride corrosion
    is_winter = datetime.now().month in [11, 12, 1, 2, 3]
    salt_multiplier = 3.0 if (is_winter and temperature < 5) else 1.0
    precip_score = min(100, precipitation * 10 * salt_multiplier)
    
    # 5. Wind loading (0-100)
    # Dynamic stress on structure
    wind_score = min(100, wind_speed * 2)
    
    # 6. Age degradation factor
    # Older bridges more susceptible
    age_factor = 1.0 + (bridge_age - 50) * 0.01 if bridge_age > 50 else 1.0
    
    # Combined environmental score (weighted)
    combined = (
        freeze_thaw_score * 0.30 +  # Major factor for NY
        humidity_score * 0.20 +
        temp_score * 0.15 +
        precip_score * 0.25 +       # Salt corrosion critical
        wind_score * 0.10
    ) * age_factor
    
    combined = min(100, combined)
    
    return {
        "freeze_thaw_score": round(freeze_thaw_score, 1),
        "humidity_score": round(humidity_score, 1),
        "temperature_score": round(temp_score, 1),
        "precipitation_score": round(precip_score, 1),
        "wind_score": round(wind_score, 1),
        "age_factor": round(age_factor, 2),
        "combined": round(combined, 1)
    }


# =============================================================================
# 4) COMPUTER VISION / CAMERA PIPELINE
# =============================================================================

def capture_frame(stream_url: str, timeout: int = 10) -> Optional[np.ndarray]:
    """Capture frame from HLS stream or YouTube using ffmpeg"""
    try:
        # Check if it's a YouTube URL
        if 'youtube.com' in stream_url or 'youtu.be' in stream_url:
            try:
                # Use yt-dlp to get the direct stream URL
                yt_dlp_cmd = [
                    'yt-dlp',
                    '-f', 'best[ext=mp4]',  # Get best quality mp4
                    '-g',  # Get URL only
                    stream_url
                ]
                yt_result = subprocess.run(yt_dlp_cmd, capture_output=True, timeout=10, text=True)
                
                if yt_result.returncode == 0 and yt_result.stdout.strip():
                    stream_url = yt_result.stdout.strip().split('\n')[0]  # Get first URL
                    logger.info(f"Resolved YouTube URL to: {stream_url[:100]}...")
                else:
                    logger.error(f"yt-dlp failed: {yt_result.stderr}")
                    return None
            except FileNotFoundError:
                logger.error("yt-dlp not found. Please install: pip install yt-dlp")
                return None
            except Exception as e:
                logger.error(f"yt-dlp error: {e}")
                return None
        
        # Use ffmpeg to capture frame
        cmd = [
            'ffmpeg', '-loglevel', 'error',
            '-i', stream_url,
            '-vframes', '1',
            '-f', 'image2pipe',
            '-vcodec', 'png', '-'
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        
        if result.returncode == 0 and result.stdout:
            image = Image.open(io.BytesIO(result.stdout))
            frame = np.array(image)
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        else:
            logger.warning(f"ffmpeg returned code {result.returncode}: {result.stderr.decode()[:200]}")
            return None
    except subprocess.TimeoutExpired:
        logger.error(f"Frame capture timed out after {timeout}s")
        return None
    except FileNotFoundError:
        logger.error("ffmpeg not found. Please install ffmpeg.")
        return None
    except Exception as e:
        logger.error(f"Frame capture error: {type(e).__name__}: {e}")
        return None

def compute_fatigue_damage(stress_series: list, m: int = 3, C: float = 1e12) -> Tuple[float, float, float]:
    """
    Miner damage using cycle ranges from rainflow-lite.
    stress_series can be proxy stress (%) or physical stress (MPa) — but keep consistent.
    """
    if not stress_series:
        return 0.0, 0.0, 0.0

    s = np.asarray(stress_series, dtype=float)
    mean_stress = float(np.mean(s))
    std_stress = float(np.std(s))

    ranges = rainflow_ranges(stress_series)
    if ranges.size == 0:
        return 0.0, mean_stress, std_stress

    # avoid zeros
    ranges = np.clip(ranges, 1e-9, None)

    # Miner: D = Σ (1/Ni), Ni = C / (ΔS^m)
    Ni = C / (ranges ** m)
    damage = float(np.sum(1.0 / np.clip(Ni, 1e-12, None)))

    return damage, mean_stress, std_stress


def compute_reliability_index(
    mu_R: float = 250.0,
    sigma_R: float = 25.0,
    mu_S: float = 0.0,
    sigma_S: float = 0.0
) -> float:
    """
    Compute reliability index β = (μ_R - μ_S) / sqrt(σ_R² + σ_S²)
    - Default resistance: μ_R = 250 MPa, σ_R = 25 MPa (10% CoV)
    - Load effect μ_S, σ_S from LWR + Miner damage
    """
    denominator = np.sqrt(sigma_R ** 2 + sigma_S ** 2)
    if denominator == 0:
        return 0.0
    return (mu_R - mu_S) / denominator

def compute_validation_metrics(detected_hourly: pd.DataFrame, nysdot_hourly: pd.DataFrame) -> Dict:
    """
    Inputs must have:
      detected_hourly: columns ["hour", "detected"]
      nysdot_hourly:   columns ["hour", "nysdot"]
    """
    df = detected_hourly.merge(nysdot_hourly, on="hour", how="inner").copy()
    if df.empty:
        return {"ok": False, "reason": "No overlapping hours", "rows": 0}

    df["error"] = df["detected"] - df["nysdot"]
    df["abs_error"] = df["error"].abs()
    df["ape"] = np.where(df["nysdot"] > 0, df["abs_error"] / df["nysdot"] * 100.0, np.nan)

    mae = float(df["abs_error"].mean())
    mape = float(np.nanmean(df["ape"])) if np.isfinite(np.nanmean(df["ape"])) else np.nan
    bias = float(df["error"].mean())
    rmse = float(np.sqrt(np.mean(df["error"] ** 2)))

    return {
        "ok": True,
        "rows": int(len(df)),
        "MAE": mae,
        "MAPE_pct": mape,
        "Bias": bias,
        "RMSE": rmse,
        "hourly_table": df[["hour", "detected", "nysdot", "error", "ape"]]
    }

def aggregate_session_hourly(session_log: list) -> pd.DataFrame:
    """
    session_log entries expected to contain:
      entry["timestamp"] (datetime)
      entry["vehicle_data"]["approaching"]["total"] OR pick the lane you want
    """
    if not session_log:
        return pd.DataFrame(columns=["hour", "detected"])

    df = pd.DataFrame([{
        "timestamp": e["timestamp"],
        "hour": e["timestamp"].strftime("%H:00"),
        "detected": e["vehicle_data"]["approaching"]["total"]
    } for e in session_log])

    out = df.groupby("hour", as_index=False)["detected"].sum()
    if isinstance(out, pd.Series):
        out = out.to_frame().reset_index()
    return out

def rainflow_ranges(series: list) -> np.ndarray:
    """
    Lightweight rainflow-ish range extraction using turning points.
    Returns an array of cycle ranges.
    This is not a full ASTM implementation, but far better than max-min.
    """
    if series is None or len(series) < 6:
        return np.array([])

    x = np.asarray(series, dtype=float)

    # turning points (peaks/valleys)
    dx = np.diff(x)
    sign = np.sign(dx)
    sign[sign == 0] = 1
    turn = np.where(np.diff(sign) != 0)[0] + 1
    tp = np.concatenate(([0], turn, [len(x) - 1]))
    y = x[tp]

    stack = []
    ranges = []

    for v in y:
        stack.append(v)
        while len(stack) >= 3:
            s0, s1, s2 = stack[-3], stack[-2], stack[-1]
            r1 = abs(s1 - s0)
            r2 = abs(s2 - s1)
            if r2 < r1:
                break
            # closed cycle at s1
            ranges.append(r1)
            stack.pop(-2)

    # remaining half-cycles
    for i in range(len(stack) - 1):
        ranges.append(abs(stack[i + 1] - stack[i]))

    return np.asarray(ranges, dtype=float)


def detect_vehicles(
    frame: np.ndarray,
    model,
    camera_config: Dict,
    camera_name: str,
    lane_divider: float = 0.43,
    confidence: float = 0.15,
    bridge_config: Optional[BridgeConfig] = None,
    use_roi: bool = True,
    roi_override: Optional[Dict] = None,
    weights: Optional[Dict[str, float]] = None,
    focal_length_px: Optional[float] = None,
) -> Tuple[Dict, np.ndarray, list]:
    """
    Detect vehicles with ROI box and distance estimation.
    
    Returns:
        - vehicle_data: counts and statistics
        - output_frame: annotated frame
        - detections: list of individual detection details (for analysis)
    """
    
    if frame is None or model is None:
        return {}, frame, []
    
    try:
        output_frame = frame.copy()
        height, width = frame.shape[:2]
        approaching_side = camera_config["approaching_side"]
        
        # Get ROI
        if roi_override:
            roi_pct = roi_override
        elif camera_name in DEFAULT_ROI:
            roi_pct = DEFAULT_ROI[camera_name]
        else:
            # Fallback to full frame with divider
            roi_pct = {"x1_pct": 0.0, "y1_pct": 0.0, "x2_pct": 1.0, "y2_pct": 1.0}
        
        roi_pixels = get_roi_pixels(frame.shape, roi_pct)
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_pixels
        
        # Draw ROI box (blue, semi-transparent effect via dashed line)
        if use_roi:
            # Draw ROI rectangle
            cv2.rectangle(output_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 165, 0), 2)
            cv2.putText(output_frame, "ROI", (roi_x1 + 5, roi_y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        # Lane divider (within ROI)
        divider_x = int(width * lane_divider)
        cv2.line(output_frame, (divider_x, roi_y1), (divider_x, roi_y2), (0, 200, 255), 2)
        
        # Initialize counts
        vehicle_data = {
            "approaching": {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "total": 0},
            "leaving": {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "total": 0},
            "in_roi": {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "total": 0},
            "outside_roi": 0,
        }
        
        detections = []  # Store individual detections for analysis
        
        # Run YOLO
        results = model(frame, conf=confidence, iou=0.45, verbose=False, classes=[2, 3, 5, 7])
        
        # Process detections
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id not in VEHICLE_CLASSES:
                    continue
                
                vehicle_type = VEHICLE_CLASSES[class_id]
                conf_score = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                
                # Check if in ROI
                in_roi = is_in_roi(center_x, center_y, roi_pixels) if use_roi else True
                
                # Estimate distance
                dist_info = estimate_vehicle_distance(
                    bbox_width,
                    width,
                    vehicle_type,
                    focal_length_px=focal_length_px,
                )
                
                # Store detection details
                detection = {
                    "type": vehicle_type,
                    "confidence": round(conf_score, 2),
                    "bbox": (x1, y1, x2, y2),
                    "center": (center_x, center_y),
                    "bbox_width_px": bbox_width,
                    "bbox_height_px": bbox_height,
                    "distance_m": dist_info["distance_m"],
                    "in_roi": in_roi,
                    "side": "left" if center_x < divider_x else "right"
                }
                detections.append(detection)
                
                if not in_roi:
                    vehicle_data["outside_roi"] += 1
                    # Draw gray box for outside ROI
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    continue
                
                # Count in ROI
                vehicle_data["in_roi"][vehicle_type] += 1
                vehicle_data["in_roi"]["total"] += 1
                
                # Determine direction
                if center_x < divider_x:
                    category = "approaching" if approaching_side == "left" else "leaving"
                else:
                    category = "approaching" if approaching_side == "right" else "leaving"
                
                vehicle_data[category][vehicle_type] += 1
                vehicle_data[category]["total"] += 1
                
                # Draw bounding box
                if category == "approaching":
                    color = (0, 255, 0)  # Green - loading bridge
                else:
                    color = (128, 128, 128)  # Gray - leaving
                
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                
                # Label with distance
                label = f"{vehicle_type[:3].upper()} {dist_info['distance_m']}m"
                cv2.putText(output_frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        w= weights or VEHICLE_WEIGHTS

        # Calculate load and density (only from ROI)
        w = weights or VEHICLE_WEIGHTS

        load_lbs = sum(
            vehicle_data["in_roi"][vtype] * float(w.get(vtype, 0))
            for vtype in ["car", "truck", "bus", "motorcycle"]
        )
        vehicle_data["load_tons"] = round(load_lbs / 2000.0, 2)
        
        # Density calculation
        roi_width_m = bridge_config.total_length_m if bridge_config else 237.4
        vehicle_data["density"] = round(vehicle_data["in_roi"]["total"] / roi_width_m, 4)
        # Ensure density exists in ONE canonical variable
        density_veh_per_m = float(vehicle_data.get("density", 0.03) or 0.03)

        # Store both keys for backward compatibility (so nothing else breaks)
        vehicle_data["density_veh_per_m"] = density_veh_per_m
        vehicle_data["density"] = density_veh_per_m

        
        # Detection statistics
        if detections:
            distances = [d["distance_m"] for d in detections if d["distance_m"] < 500]
            vehicle_data["detection_stats"] = {
                "total_detected": len(detections),
                "in_roi": vehicle_data["in_roi"]["total"],
                "outside_roi": vehicle_data["outside_roi"],
                "min_distance_m": round(min(distances), 1) if distances else 0,
                "max_distance_m": round(max(distances), 1) if distances else 0,
                "avg_distance_m": round(np.mean(distances), 1) if distances else 0,
            }
        else:
            vehicle_data["detection_stats"] = {
                "total_detected": 0, "in_roi": 0, "outside_roi": 0,
                "min_distance_m": 0, "max_distance_m": 0, "avg_distance_m": 0
            }
        
        return vehicle_data, output_frame, detections
    
    except Exception as e:
        logger.error(f"Detection error: {type(e).__name__}: {e}")
        return {}, frame, []


def estimate_avg_speed_mps(
    detections: list,
    prev_detections: list,
    dt_s: float,
    max_match_dist_px: float = 200.0,
    dt_min_s: Optional[float] = None,
    dt_max_s: Optional[float] = None,
    deadband_m: float = 0.0,
) -> Optional[float]:
    """
    Estimate average speed (m/s) by matching detections across frames.
    Uses nearest-center matching within a pixel threshold, same type and side.
    """
    if not detections or not prev_detections or dt_s <= 0:
        return None

    if dt_min_s is not None and dt_s < dt_min_s:
        dt_s = dt_min_s
    if dt_max_s is not None and dt_s > dt_max_s:
        dt_s = dt_max_s
    if dt_s <= 0:
        return None

    max_match_dist_sq = float(max_match_dist_px) * float(max_match_dist_px)
    speeds = []

    for det in detections:
        if det.get("distance_m") is None:
            continue
        best = None
        best_dist_sq = None
        for prev in prev_detections:
            if prev.get("type") != det.get("type"):
                continue
            if prev.get("side") != det.get("side"):
                continue
            dx = det["center"][0] - prev["center"][0]
            dy = det["center"][1] - prev["center"][1]
            dist_sq = dx * dx + dy * dy
            if best_dist_sq is None or dist_sq < best_dist_sq:
                best = prev
                best_dist_sq = dist_sq
        if best is None or best_dist_sq is None or best_dist_sq > max_match_dist_sq:
            continue
        prev_dist = best.get("distance_m")
        if prev_dist is None:
            continue
        dist_delta = abs(det["distance_m"] - prev_dist)
        if deadband_m > 0.0 and dist_delta < deadband_m:
            speed = 0.0
        else:
            speed = dist_delta / dt_s
        if 0 <= speed <= 60:
            speeds.append(speed)

    if not speeds:
        return None
    return float(np.median(speeds))


def estimate_avg_speed_mps_pixel(
    detections: list,
    prev_detections: list,
    dt_s: float,
    meters_per_pixel: float,
    axis: str = "x",
    max_match_dist_px: float = 200.0,
    dt_min_s: Optional[float] = None,
    dt_max_s: Optional[float] = None,
    deadband_px: float = 0.0,
) -> Optional[float]:
    """
    Estimate average speed (m/s) using pixel displacement along a chosen axis.
    Axis is 'x', 'y', or 'diag' (euclidean). meters_per_pixel is a simple scale.
    """
    if not detections or not prev_detections or dt_s <= 0:
        return None
    if dt_min_s is not None and dt_s < dt_min_s:
        dt_s = dt_min_s
    if dt_max_s is not None and dt_s > dt_max_s:
        dt_s = dt_max_s
    if dt_s <= 0:
        return None

    max_match_dist_sq = float(max_match_dist_px) * float(max_match_dist_px)
    speeds = []

    for det in detections:
        best = None
        best_dist_sq = None
        for prev in prev_detections:
            if prev.get("type") != det.get("type"):
                continue
            if prev.get("side") != det.get("side"):
                continue
            dx = det["center"][0] - prev["center"][0]
            dy = det["center"][1] - prev["center"][1]
            dist_sq = dx * dx + dy * dy
            if best_dist_sq is None or dist_sq < best_dist_sq:
                best = prev
                best_dist_sq = dist_sq
        if best is None or best_dist_sq is None or best_dist_sq > max_match_dist_sq:
            continue

        dx = det["center"][0] - best["center"][0]
        dy = det["center"][1] - best["center"][1]
        if axis == "y":
            disp_px = abs(dy)
        elif axis == "diag":
            disp_px = float(np.hypot(dx, dy))
        else:
            disp_px = abs(dx)

        if deadband_px > 0.0 and disp_px < deadband_px:
            speed = 0.0
        else:
            speed = (disp_px * float(meters_per_pixel)) / dt_s
        if 0 <= speed <= 60:
            speeds.append(speed)

    if not speeds:
        return None
    return float(np.median(speeds))

def load_tons_to_stress_mpa(load_tons: float, k_mpa_per_ton: float = 0.6) -> float:
    """
    Simple surrogate: stress (MPa) = k * load_tons
    Tune k using any known calibration point (or keep as proxy and say so).
    """
    return float(max(0.0, load_tons) * k_mpa_per_ton)

# =============================================================================
# 5) SIMULATION + ML
# =============================================================================

def run_lwr_simulation(
    initial_density: float,          # vehicles/m  (NOT a ratio)
    road_length_m: float,
    v_max_mps: float,
    total_time: int = 300,
    inject_jam: bool = False,
    rho_max: float = 0.20            # vehicles/m (jam density cap)
) -> Dict:
    """LWR traffic flow simulation (density in vehicles/m)."""

    dx, dt = 10.0, 1.0
    num_sections = max(3, int(road_length_m / dx))
    num_steps = max(1, int(total_time / dt))

    # density field in vehicles/m
    rho = np.ones(num_sections) * float(initial_density)
    rho += np.random.normal(0, 0.01 * rho_max, num_sections)
    rho = np.clip(rho, 0.0, rho_max)

    stress_history = []

    for step in range(num_steps):
        # Greenshields: v = v_max (1 - rho/rho_max)
        velocity = v_max_mps * (1.0 - rho / rho_max)
        velocity = np.clip(velocity, 0.0, v_max_mps)

        flow = rho * velocity
        wave_speed = v_max_mps * (1.0 - 2.0 * rho / rho_max)

        rho_new = rho.copy()
        for i in range(1, num_sections - 1):
            if wave_speed[i] > 0:
                rho_new[i] = rho[i] - (dt / dx) * (flow[i] - flow[i - 1])
            else:
                rho_new[i] = rho[i] - (dt / dx) * (flow[i + 1] - flow[i])

        # upstream boundary: keep mean close to initial_density (vehicles/m)
        rho_new[0] = float(initial_density)

        rho = np.clip(rho_new, 0.0, rho_max)

        # probabilistic jam injection mid-sim
        if inject_jam and step == num_steps // 2:
            mid = num_sections // 2
            jam_width = np.random.randint(5, 15)
            rho[mid:mid + jam_width] = rho_max * np.random.uniform(0.70, 0.95)

        # stress proxy as % of rho_max (kept as proxy)
        stress = float(np.mean(rho) / rho_max * 100.0)
        stress_history.append(stress)

    cumulative = float(np.trapezoid(stress_history, dx=dt))
    fatigue = float(np.clip(cumulative / 100.0, 0.0, 100.0))

    shockwave_speed = float(np.mean(np.abs(np.gradient(rho))) * v_max_mps)

    return {
        "fatigue": fatigue,
        "shockwave_speed": shockwave_speed,
        "avg_density": float(np.mean(rho)),
        "max_stress": float(max(stress_history)) if stress_history else 0.0,
        "stress_history": stress_history
    }

def plot_shockwave_sim_vs_obs(session_log: list) -> "go.Figure":
    times = [e["timestamp"].strftime("%H:%M:%S") for e in session_log]

    sim_sw = [e.get("sim_shockwave", None) for e in session_log]
    obs_sw = [e.get("obs_shockwave", None) for e in session_log]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=sim_sw, mode="lines+markers", name="Sim shockwave (LWR)"))
    fig.add_trace(go.Scatter(x=times, y=obs_sw, mode="lines+markers", name="Observed shockwave (proxy)"))

    fig.update_layout(
        title="Shockwave Speed (Sim vs Observed)",
        xaxis_title="Capture Time",
        yaxis_title="Shockwave speed (m/s)",
        template="plotly_white",
        height=330
    )
    return fig


def plot_sim_obs_over_time(session_log: list) -> "go.Figure":
    times = [e["timestamp"].strftime("%H:%M:%S") for e in session_log]

    sim_sh = [e.get("sim_shockwave", None) for e in session_log]
    obs_sh = [e.get("obs_shockwave", None) for e in session_log]

    beta_sim = [e.get("sim_beta", None) for e in session_log]
    beta_obs = [e.get("obs_beta", None) for e in session_log]
    beta_primary = [e.get("beta_primary", None) for e in session_log]
    gap_beta = [e.get("gap_beta", None) for e in session_log]

    fig = go.Figure()

    # Shockwaves
    fig.add_trace(go.Scatter(x=times, y=sim_sh, mode="lines+markers", name="Sim Shockwave (LWR)"))
    fig.add_trace(go.Scatter(x=times, y=obs_sh, mode="lines+markers", name="Obs Shockwave (proxy)"))

    # Reliability betas
    fig.add_trace(go.Scatter(x=times, y=beta_sim, mode="lines+markers", name="β Sim"))
    fig.add_trace(go.Scatter(x=times, y=beta_obs, mode="lines+markers", name="β Obs"))
    fig.add_trace(go.Scatter(x=times, y=beta_primary, mode="lines", name="β Primary", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=times, y=gap_beta, mode="lines", name="Gap β (Obs-Sim)", visible="legendonly"))

    fig.add_hline(y=3.0, line_dash="dash", annotation_text="Target β=3.0")

    fig.update_layout(
        title="Sim vs Observed: Shockwave and Reliability Over Time",
        xaxis_title="Capture Time",
        yaxis_title="Value (mixed units)",
        template="plotly_white",
        height=460,
        legend=dict(orientation="h")
    )

    return fig


def plot_reliability_over_time(session_log: list) -> "go.Figure":
    times = [e["timestamp"].strftime("%H:%M:%S") for e in session_log]

    sim_b = [e.get("sim_beta", None) for e in session_log]
    obs_b = [e.get("obs_beta", None) for e in session_log]
    gap_b = [e.get("gap_beta", None) for e in session_log]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=sim_b, mode="lines+markers", name="Sim β (LWR→Miner)"))
    fig.add_trace(go.Scatter(x=times, y=obs_b, mode="lines+markers", name="Obs β (Camera→Miner)"))
    fig.add_trace(go.Scatter(x=times, y=gap_b, mode="lines", name="Gap β (Obs - Sim)", visible="legendonly"))

    fig.add_hline(y=3.0, line_dash="dash", annotation_text="Target β=3.0")
    fig.update_layout(
        title="Reliability Index Over Time (Sim vs Observed)",
        xaxis_title="Capture Time",
        yaxis_title="β",
        template="plotly_white",
        height=380
    )
    return fig


def run_monte_carlo(
    num_runs: int,
    road_length_m: float,
    live_density: Optional[float] = None,     # vehicles/m
    inject_jam_probability: float = 0.3,
    rho_max: float = 0.20                      # vehicles/m
) -> pd.DataFrame:
    """
    Monte Carlo simulation with density in vehicles/m.
    If live_density is provided, runs sample around it.
    """

    results = []

    for _ in range(num_runs):
        if live_density is not None:
            if live_density < 1e-6:
                results.append({
                    "density": 0.0, "v_max": 80.0, "alpha": 0.0005,
                    "shockwave_speed": 0.0, "fatigue": 0.0, "jam_injected": False
                })
                continue

            # std = 10% of rho_max (tunable)
            sigma = 0.10 * rho_max
            density = float(np.random.normal(live_density, sigma))
            density = float(np.clip(density, 0.0, rho_max))
        else:
            density = float(np.random.uniform(0.02, 0.12))  # vehicles/m typical range

        v_max = float(np.random.choice([40, 60, 80, 100]) / 3.6)  # m/s
        alpha = float(np.random.uniform(0.00005, 0.001))

        inject_jam = bool(np.random.random() < inject_jam_probability)

        sim = run_lwr_simulation(
            initial_density=density,
            road_length_m=road_length_m,
            v_max_mps=v_max,
            inject_jam=inject_jam,
            rho_max=rho_max
        )

        results.append({
            "density": density,
            "v_max": v_max * 3.6,  # km/h
            "alpha": alpha,
            "shockwave_speed": sim["shockwave_speed"],
            "fatigue": sim["fatigue"],
            "jam_injected": inject_jam
        })

    return pd.DataFrame(results)



def train_model_cv(data: pd.DataFrame, k_folds: int = 5, seeds=(0, 1, 2)) -> Tuple:
    """
    K-fold CV across multiple seeds.
    Returns: (final_model, metrics_dict)
    """
    if not SKLEARN_AVAILABLE or len(data) < 50:
        logger.warning(f"Cannot train: sklearn={SKLEARN_AVAILABLE}, samples={len(data)}")
        return None, {}

    traffic_features = ["density", "v_max", "shockwave_speed", "truck_pct"]
    weather_features = ["temperature", "temp_range", "precipitation", "wind_speed",
                        "freeze_thaw", "salt_exposure", "month", "is_winter"]

    features = [f for f in (traffic_features + weather_features) if f in data.columns]
    if len(features) < 4:
        return None, {"error": f"Not enough features: {features}"}

    X = data[features].copy()
    y = data["fatigue"].astype(float).copy()

    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.ensemble import RandomForestRegressor

    fold_results = []
    importance_accum = {f: [] for f in features}

    for seed in seeds:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        for tr_idx, te_idx in kf.split(X):
            Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
            ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=18,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1
            )
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)

            fold_results.append({
                "seed": seed,
                "r2": float(r2_score(yte, pred)),
                "mae": float(mean_absolute_error(yte, pred))
            })

            for f, imp in zip(features, model.feature_importances_):
                importance_accum[f].append(float(imp))

    folds_df = pd.DataFrame(fold_results)
    importance_mean = {f: float(np.mean(v)) for f, v in importance_accum.items()}
    importance_std = {f: float(np.std(v)) for f, v in importance_accum.items()}

    # Train final model on all data (pick fixed seed)
    final_model = RandomForestRegressor(
        n_estimators=400,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X, y)

    metrics = {
        "cv_r2_mean": float(folds_df["r2"].mean()),
        "cv_r2_std": float(folds_df["r2"].std()),
        "cv_mae_mean": float(folds_df["mae"].mean()),
        "cv_mae_std": float(folds_df["mae"].std()),
        "feature_importance_mean": dict(sorted(importance_mean.items(), key=lambda x: x[1], reverse=True)),
        "feature_importance_std": importance_std,
        "features_used": features,
        "samples": int(len(data)),
        "folds_df": folds_df
    }

    return final_model, metrics



def predict_fatigue_with_weather(
    model,
    density: float,
    shockwave: float,
    truck_pct: float,
    temperature: float,
    precipitation: float,
    wind_speed: float,
    freeze_thaw: int,
    month: int
) -> Tuple[float, Dict]:
    """
    Predict fatigue using BOTH current traffic AND current weather.
    
    This is real-time inference using the trained model.
    """
    
    if model is None:
        # Fallback to simple calculation
        return density * 100, {"method": "fallback"}
    
    # Derived features
    is_winter = 1 if month in [11, 12, 1, 2, 3] else 0
    temp_range = 10  # Assumed average daily range
    salt_exposure = precipitation * 3.0 if (is_winter and temperature < 5) else precipitation * 0.5
    
    # Build feature vector
    features = {
        "density": density,
        "v_max": 80,  # Default assumption
        "shockwave_speed": shockwave,
        "truck_pct": truck_pct,
        "temperature": temperature,
        "temp_range": temp_range,
        "precipitation": precipitation,
        "wind_speed": wind_speed,
        "freeze_thaw": freeze_thaw,
        "salt_exposure": salt_exposure,
        "month": month,
        "is_winter": is_winter
    }
    
    # Create DataFrame with same columns as training
    X = pd.DataFrame([features])
    
    # Get available features (model might have been trained with subset)
    try:
        model_features = model.feature_names_in_
        X = X[model_features]
    except AttributeError:
        pass
    
    prediction = model.predict(X)[0]
    
    logger.info(f"PREDICTION - Density: {density:.4f}, Temp: {temperature}°C, "
                f"Precip: {precipitation}mm, F/T: {freeze_thaw} → Fatigue: {prediction:.1f}")
    
    return prediction, features

def compute_confidence_score(vehicle_data: Dict, weather: Dict, rf_metrics: Dict) -> Dict:
    """
    Returns a confidence score (0-1), label, and reasons.
    Uses only signals already available in the app.
    """
    stats = vehicle_data.get("detection_stats", {})
    total_detected = float(stats.get("total_detected", 0) or 0)
    in_roi = float(stats.get("in_roi", 0) or 0)
    roi_ratio = (in_roi / total_detected) if total_detected > 0 else 0.0

    speed_ok = 1.0 if vehicle_data.get("avg_speed_mps") is not None else 0.0
    weather_ok = 1.0 if weather.get("success", False) else 0.0

    # RF quality proxy if available
    r2 = rf_metrics.get("cv_r2_mean", None)
    if r2 is None:
        rf_ok = 0.0
        rf_strength = 0.0
    else:
        rf_ok = 1.0
        rf_strength = float(np.clip((r2 + 1.0) / 2.0, 0.0, 1.0))  # map [-1,1] -> [0,1]

    # Detection sufficiency
    det_strength = float(np.clip(total_detected / 10.0, 0.0, 1.0))  # >=10 detections => strong
    roi_strength = float(np.clip(roi_ratio / 0.70, 0.0, 1.0))       # >=70% in ROI => strong

    score = (
        0.35 * det_strength +
        0.25 * roi_strength +
        0.15 * speed_ok +
        0.10 * weather_ok +
        0.15 * rf_strength
    )
    score = float(np.clip(score, 0.0, 1.0))

    if score >= 0.75:
        label = "High"
    elif score >= 0.50:
        label = "Medium"
    else:
        label = "Low"

    reasons = []
    reasons.append(f"Detections: {int(total_detected)}")
    reasons.append(f"ROI coverage: {roi_ratio*100:.0f}%")
    reasons.append("Speed: OK" if speed_ok else "Speed: N/A")
    reasons.append("Weather: OK" if weather_ok else "Weather: fallback")
    if rf_ok:
        reasons.append(f"RF CV R²: {r2:.2f}")
    else:
        reasons.append("RF: not trained")

    return {"score": round(score, 2), "label": label, "reasons": reasons}


def detect_events(session_log: list, speed_drop_pct: float = 30.0) -> list:
    """
    Create event tags based on recent captures.
    Returns a list of strings (event codes).
    """
    if not session_log:
        return []

    latest = session_log[-1]
    vd = latest.get("vehicle_data", {})
    approaching = vd.get("approaching", {})
    total = float(approaching.get("total", 0) or 0)
    trucks = float(approaching.get("truck", 0) or 0)
    truck_pct = (trucks / total) if total > 0 else 0.0

    density = float(vd.get("density", 0.0) or 0.0)
    speed = vd.get("avg_speed_kph", None)

    events = []

    # Truck surge
    if truck_pct >= 0.25 and total >= 5:
        events.append("TRUCK_SURGE")

    # Jam likely: high density + low speed
    if speed is not None and density >= 0.08 and speed <= 25:
        events.append("JAM_LIKELY")

    # Speed drop compared to previous
    if len(session_log) >= 2:
        prev = session_log[-2].get("vehicle_data", {})
        prev_speed = prev.get("avg_speed_kph", None)
        if speed is not None and prev_speed is not None and prev_speed > 1:
            drop = (prev_speed - speed) / prev_speed * 100.0
            if drop >= speed_drop_pct:
                events.append("SPEED_DROP_EVENT")

    # Heavy load event
    load_tons = float(vd.get("load_tons", 0.0) or 0.0)
    if load_tons >= 60:
        events.append("HEAVY_LOAD")

    return events


def compute_sim_obs_gap(entry: Dict) -> Dict:
    """
    Compute gaps between observed and simulated outputs.
    """
    sim_damage = float(entry.get("sim_damage", 0.0) or 0.0)
    obs_damage = float(entry.get("obs_damage", 0.0) or 0.0)
    sim_beta = entry.get("sim_beta", None)
    obs_beta = entry.get("obs_beta", None)

    gap_damage = obs_damage - sim_damage
    gap_beta = None
    if sim_beta is not None and obs_beta is not None:
        gap_beta = float(obs_beta) - float(sim_beta)

    return {"gap_damage": round(gap_damage, 6), "gap_beta": round(gap_beta, 2) if gap_beta is not None else None}


def predict_fatigue(
    model,
    density: float,
    avg_shockwave: float
) -> float:
    """Predict traffic fatigue from current conditions"""
    if density < 0.005:
        return 0.0  # No traffic → no fatigue
    
    if model is None:
        return density * 100  # Fallback
    
    X = pd.DataFrame([{
        "density": density,
        "v_max": 80,
        "alpha": 0.0005,
        "shockwave_speed": avg_shockwave
    }])
    
    return model.predict(X)[0]



# =============================================================================
# 6) SCENARIO ANALYSIS
# =============================================================================

def calculate_scenario_fatigue(
    base_traffic_fatigue: float,
    base_env_stress: float,
    traffic_multiplier: float,
    truck_percentage: float,
    freeze_thaw_cycles: int,
    temperature: float,
    precipitation: float,
    bridge_age: int
) -> Dict:
    """Calculate fatigue for a given scenario"""
    
    # Adjust traffic fatigue
    truck_factor = 1 + (truck_percentage - 0.15) * 2  # Trucks have higher impact
    if base_traffic_fatigue < 1.0:
        adjusted_traffic = 0.0
    else:
        adjusted_traffic = base_traffic_fatigue * traffic_multiplier * truck_factor
    adjusted_traffic = min(100, adjusted_traffic)
    
    # Recalculate environmental
    env_stress = calculate_environmental_stress(
        temperature=temperature,
        humidity=50,
        precipitation=precipitation,
        freeze_thaw_cycles=freeze_thaw_cycles,
        wind_speed=10,
        bridge_age=datetime.now().year - 1927
    )
    
    # Combined fatigue (weighted)
    # Traffic is primary, environment is modifier
    combined = adjusted_traffic * 0.7 + env_stress["combined"] * 0.3
    combined = min(100, combined)
    
    return {
        "traffic_fatigue": round(adjusted_traffic, 1),
        "environmental_stress": round(env_stress["combined"], 1),
        "combined_fatigue": round(combined, 1),
        "env_breakdown": env_stress
    }

def choose_vmax_from_observed_speed(avg_speed_mps: Optional[float]) -> float:
    """
    Map observed average speed to a plausible free-flow v_max for LWR.
    Keep bounded to avoid extreme values.
    """
    if avg_speed_mps is None:
        return 22.2  # fallback ~80 km/h

    # v_max should be >= observed speed, but not ridiculous
    # small uplift accounts for "free flow" vs measured average
    v = float(avg_speed_mps * 1.20)
    return float(np.clip(v, 8.0, 33.0))  # 8 m/s (29 km/h) to 33 m/s (119 km/h)


def get_status(score: float) -> Tuple[str, str]:
    """Get status label and color from fatigue score"""
    if score < 50:
        return "✅ SAFE OPERATION", "green"
    elif score < 70:
        return "⚠️ MONITOR CLOSELY", "orange"
    elif score < 85:
        return "🔶 SCHEDULE INSPECTION", "red"
    else:
        return "🔴 IMMEDIATE ACTION", "darkred"


# =============================================================================
# 7) VISUALIZATIONS
# =============================================================================

def create_fatigue_breakdown_chart(traffic: float, environmental: float) -> go.Figure:
    """Stacked bar showing fatigue components"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=["Fatigue Score"],
        y=[traffic * 0.7],
        name="Traffic Stress (70%)",
        marker_color="#3498db"
    ))
    
    fig.add_trace(go.Bar(
        x=["Fatigue Score"],
        y=[environmental * 0.3],
        name="Environmental (30%)",
        marker_color="#e74c3c"
    ))
    
    fig.update_layout(
        barmode="stack",
        height=300,
        title="Fatigue Score Breakdown",
        yaxis_title="Score",
        showlegend=True,
        template="plotly_white"
    )
    
    return fig


def create_sensitivity_chart(base_traffic: float, base_env: float) -> go.Figure:
    """Show how fatigue changes with different scenarios"""
    
    traffic_mult = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
    scores = []
    
    for mult in traffic_mult:
        adj_traffic = base_traffic * mult
        combined = adj_traffic * 0.7 + base_env * 0.3
        scores.append(min(100, combined))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=[f"{int((m-1)*100):+d}%" for m in traffic_mult],
        y=scores,
        mode="lines+markers",
        name="Combined Fatigue",
        line=dict(color="#2ecc71", width=3)
    ))
    
    # Threshold lines
    fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Safe")
    fig.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Monitor")
    fig.add_hline(y=85, line_dash="dash", line_color="red", annotation_text="Critical")
    
    fig.update_layout(
        height=300,
        title="Sensitivity: Traffic Change Impact",
        xaxis_title="Traffic Change",
        yaxis_title="Fatigue Score",
        template="plotly_white"
    )
    
    return fig


def create_environmental_breakdown_chart(env_data: Dict) -> go.Figure:
    """Pie chart of environmental factors"""
    
    labels = ["Freeze-Thaw", "Humidity", "Temperature", "Precipitation", "Wind"]
    values = [
        env_data["freeze_thaw_score"],
        env_data["humidity_score"],
        env_data["temperature_score"],
        env_data["precipitation_score"],
        env_data["wind_score"]
    ]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=["#3498db", "#9b59b6", "#e74c3c", "#1abc9c", "#f39c12"]
    )])
    
    fig.update_layout(
        height=300,
        title="Environmental Stress Factors",
        template="plotly_white"
    )
    
    return fig


def create_historical_weather_charts(monthly_data: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """Create charts for historical weather analysis"""
    
    # Chart 1: Freeze-thaw cycles by month
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=monthly_data["year_month"],
        y=monthly_data["freeze_thaw"],
        name="Freeze-Thaw Cycles",
        marker_color="#3498db"
    ))
    fig1.update_layout(
        title="Monthly Freeze-Thaw Cycles (Last 12 Months)",
        xaxis_title="Month",
        yaxis_title="Cycles",
        template="plotly_white",
        height=350
    )
    
    # Chart 2: Salt exposure (winter precipitation)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=monthly_data["year_month"],
        y=monthly_data["salt_exposure"],
        name="Salt Exposure",
        marker_color="#e74c3c"
    ))
    fig2.add_trace(go.Scatter(
        x=monthly_data["year_month"],
        y=monthly_data["precipitation"],
        name="Total Precipitation",
        mode="lines+markers",
        line=dict(color="#2ecc71", width=2)
    ))
    fig2.update_layout(
        title="Monthly Precipitation & Salt Exposure",
        xaxis_title="Month",
        yaxis_title="mm",
        template="plotly_white",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    return fig1, fig2

def build_sim_vs_obs_fig(session_log: list) -> "go.Figure":
    df = pd.DataFrame(session_log).copy()

    # Extract series safely
    df["t"] = df["timestamp"]
    df["sim_shockwave"] = df.get("sim_shockwave", 0.0)
    df["obs_shockwave_speed"] = df.get("obs_shockwave_speed", 0.0)

    df["beta_sim"] = df.get("sim_beta", None)
    df["beta_obs"] = df.get("obs_beta", None)
    df["beta_primary"] = df.get("beta_primary", None)
    df["gap_beta"] = df.get("gap_beta", None)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Shockwave Speed: Sim vs Observed (m/s)",
            "Reliability β: Sim vs Observed vs Primary",
            "β Gap: (Observed − Simulated)",
            "Shockwave Speed Gap: (Observed − Simulated)"
        )
    )

    # (1) Shockwave speed
    fig.add_trace(go.Scatter(x=df["t"], y=df["sim_shockwave"], mode="lines+markers", name="Sim shockwave (m/s)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["t"], y=df["obs_shockwave_speed"], mode="lines+markers", name="Obs shockwave (m/s proxy)"), row=1, col=1)

    # (2) Betas
    fig.add_trace(go.Scatter(x=df["t"], y=df["beta_sim"], mode="lines+markers", name="β_sim"), row=1, col=2)
    fig.add_trace(go.Scatter(x=df["t"], y=df["beta_obs"], mode="lines+markers", name="β_obs"), row=1, col=2)
    fig.add_trace(go.Scatter(x=df["t"], y=df["beta_primary"], mode="lines+markers", name="β_primary"), row=1, col=2)

    # (3) gap_beta
    fig.add_trace(go.Scatter(x=df["t"], y=df["gap_beta"], mode="lines+markers", name="gap_beta"), row=2, col=1)

    # (4) shockwave speed gap
    df["gap_shockwave_speed"] = df["obs_shockwave_speed"] - df["sim_shockwave"]
    fig.add_trace(go.Scatter(x=df["t"], y=df["gap_shockwave_speed"], mode="lines+markers", name="gap_shockwave_speed"), row=2, col=2)

    fig.update_layout(height=650, template="plotly_white", showlegend=True)
    return fig


def create_seasonal_fatigue_chart(seasonal_data: pd.DataFrame) -> go.Figure:
    """Create seasonal contribution chart"""
    
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    colors = ["#3498db", "#2ecc71", "#f1c40f", "#e67e22"]
    
    # Ensure all seasons present
    for s in seasons:
        if s not in seasonal_data.index:
            seasonal_data.loc[s] = 0
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name="Freeze-Thaw",
        x=seasons,
        y=[seasonal_data.loc[s, "freeze_thaw"] if s in seasonal_data.index else 0 for s in seasons],
        marker_color="#3498db"
    ))
    
    fig.add_trace(go.Bar(
        name="Salt Exposure",
        x=seasons,
        y=[seasonal_data.loc[s, "salt_exposure"] if s in seasonal_data.index else 0 for s in seasons],
        marker_color="#e74c3c"
    ))
    
    fig.update_layout(
        title="Seasonal Environmental Stress Distribution",
        xaxis_title="Season",
        yaxis_title="Cumulative Value",
        barmode="group",
        template="plotly_white",
        height=350
    )
    
    return fig

def build_observed_stress_history(
    load_tons: float,
    density: float,
    avg_speed_mps: Optional[float],
    window_s: int = 60,
    dt_s: int = 1
) -> list:
    """
    Build an 'observed' stress history (proxy) using ONLY camera-derived quantities:
    - load_tons: from YOLO class counts × weights
    - density: vehicles/m from ROI count / bridge length
    - avg_speed_mps: estimated from frame-to-frame distance deltas (if available)

    We construct a short time window stress series (window_s) so Miner can run on something.
    This is NOT structural MPa; it's a normalized proxy (0-100).
    """

    # Normalize components into 0-100-ish proxy scores
    # Tune these reference values for your bridge/demo
    load_ref_tons = 80.0       # typical high live load in tons (proxy)
    density_ref = 0.12         # vehicles/m near congestion (proxy)
    speed_free = 22.2          # ~80 km/h in m/s

    load_score = np.clip((load_tons / load_ref_tons) * 100.0, 0.0, 120.0)
    dens_score = np.clip((density / density_ref) * 100.0, 0.0, 120.0)

    if avg_speed_mps is None:
        speed_penalty = 15.0   # unknown speed → modest penalty
    else:
        # slower traffic => more stop/go => more stress cycling (proxy)
        speed_ratio = np.clip(avg_speed_mps / speed_free, 0.0, 1.5)
        speed_penalty = (1.0 - np.clip(speed_ratio, 0.0, 1.0)) * 40.0  # up to +40

    base = 0.45 * load_score + 0.45 * dens_score + 0.10 * speed_penalty
    base = float(np.clip(base, 0.0, 100.0))

    # Add small oscillation + noise to mimic stop-go cycling for Miner
    stress_series = []
    for t in range(0, window_s, dt_s):
        cyc = 6.0 * np.sin(2 * np.pi * t / 18.0)     # ~18s cycle
        noise = float(np.random.normal(0.0, 1.5))    # small jitter
        stress_series.append(float(np.clip(base + cyc + noise, 0.0, 100.0)))

    return stress_series

def proxy_stress_to_mpa(stress_proxy: list, stress_ref_mpa: float) -> list:
    """
    stress_proxy is 0–100, map to 0–stress_ref_mpa MPa (proxy).
    """
    if not stress_proxy:
        return []
    scale = float(stress_ref_mpa) / 100.0
    return [float(s) * scale for s in stress_proxy]


def compute_simulated_damage_beta(
    density: float,
    road_length_m: float,
    v_max_mps: float,
    inject_jam: bool = False,
    miner_m: int = 3,
    miner_C: float = 1e12,
    mu_R: float = 250.0,
    sigma_R: float = 25.0,
    stress_ref_mpa: float = 80.0
) -> Dict:
    sim = run_lwr_simulation(
        initial_density=density,
        road_length_m=road_length_m,
        v_max_mps=v_max_mps,
        inject_jam=inject_jam
    )
    stress_mpa_series = proxy_stress_to_mpa(sim["stress_history"], stress_ref_mpa)
    damage, mu_S, sigma_S = compute_fatigue_damage(stress_mpa_series, m=miner_m, C=miner_C)
    beta = compute_reliability_index(mu_R=mu_R, sigma_R=sigma_R, mu_S=mu_S, sigma_S=sigma_S)

    return {
        "sim": sim,
        "sim_damage": round(damage, 6),
        "sim_mu_S": round(mu_S, 4),
        "sim_sigma_S": round(sigma_S, 4),
        "sim_beta": round(beta, 2),
        "sim_shockwave": float(sim.get("shockwave_speed", 0.0)),
    }


def compute_observed_damage_beta(
    load_tons: float,
    density: float,
    avg_speed_mps: Optional[float],
    miner_m: int = 3,
    miner_C: float = 1e12,
    mu_R: float = 250.0,
    sigma_R: float = 25.0,
    stress_ref_mpa: float = 80.0
) -> Dict:
    obs_series = build_observed_stress_history(
        load_tons=load_tons,
        density=density,
        avg_speed_mps=avg_speed_mps
    )

    stress_mpa_series = proxy_stress_to_mpa(obs_series, stress_ref_mpa)
    damage, mu_S, sigma_S = compute_fatigue_damage(stress_mpa_series, m=miner_m, C=miner_C)
    beta = compute_reliability_index(mu_R=mu_R, sigma_R=sigma_R, mu_S=mu_S, sigma_S=sigma_S)

    return {
        "obs_stress_history": obs_series,
        "obs_damage": round(damage, 6),
        "obs_mu_S": round(mu_S, 4),
        "obs_sigma_S": round(sigma_S, 4),
        "obs_beta": round(beta, 2),
    }


def compute_observed_shockwave_intensity(session_log: list) -> float:
    """
    Observed shockwave intensity:
    - increases when speed drops sharply (deceleration / stop-go)
    - increases when density increases quickly
    Uses only logged observed values (density, avg_speed_mps).
    """
    if len(session_log) < 2:
        return 0.0

    cur = session_log[-1].get("vehicle_data", {})
    prev = session_log[-2].get("vehicle_data", {})

    rho_now = float(cur.get("density", 0.0) or 0.0)
    rho_prev = float(prev.get("density", 0.0) or 0.0)

    v_now = cur.get("avg_speed_mps", None)
    v_prev = prev.get("avg_speed_mps", None)

    # Density change (vehicles/m)
    drho = max(0.0, rho_now - rho_prev)

    # Speed drop (m/s)
    if v_now is None or v_prev is None:
        dv_drop = 0.0
    else:
        dv_drop = max(0.0, float(v_prev) - float(v_now))

    # Normalize to reasonable ranges (tunable)
    drho_ref = 0.02   # density jump reference
    dv_ref = 5.0      # speed drop reference (~18 km/h)

    rho_term = min(1.0, drho / drho_ref)
    v_term = min(1.0, dv_drop / dv_ref)

    # Combine (weights reflect that speed drop is the strongest stop-go signature)
    shock_proxy = 0.35 * rho_term + 0.65 * v_term

    return float(round(shock_proxy, 3))

def compute_observed_shockwave_speed_proxy(session_log: list) -> float:
    """
    Observed shockwave SPEED proxy (m/s):
    Uses recent (density change, speed drop) to estimate a wave-speed-like quantity.

    This is still a proxy (camera-derived), but it's in m/s so it can be plotted vs sim_shockwave.
    """
    if len(session_log) < 2:
        return 0.0

    cur = session_log[-1].get("vehicle_data", {})
    prev = session_log[-2].get("vehicle_data", {})

    rho_now = float(cur.get("density", 0.0) or 0.0)
    rho_prev = float(prev.get("density", 0.0) or 0.0)
    drho = rho_now - rho_prev

    v_now = cur.get("avg_speed_mps", None)
    v_prev = prev.get("avg_speed_mps", None)

    if v_now is None or v_prev is None:
        return 0.0

    dv = float(v_now) - float(v_prev)

    # If density rises and speed drops -> shock forming
    if drho <= 0.0 or dv >= 0.0:
        return 0.0

    eps = 1e-6
    # “wave-speed-like” magnitude (scaled + clipped for stability)
    w = abs(dv) / max(abs(drho), eps)

    # Scale into a plausible m/s range for display (tunable)
    w_scaled = float(np.clip(w * 0.25, 0.0, 20.0))  # 0–20 m/s

    return float(round(w_scaled, 3))

# =============================================================================
# 8) REPORT GENERATION
# =============================================================================

def get_report_css() -> str:
    """Return common CSS for HTML reports"""
    return """
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; line-height: 1.6; color: #333; }
        h1 { color: #2c3e50; margin: 0; }
        h2 { color: #34495e; margin-top: 30px; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
        .header p { margin: 5px 0; opacity: 0.9; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .card { background: #f8f9fa; border-radius: 8px; padding: 20px; border-left: 4px solid #3498db; }
        .card.warning { border-left-color: #f39c12; }
        .card.danger { border-left-color: #e74c3c; }
        .card.success { border-left-color: #27ae60; }
        .card h3 { margin: 0 0 10px 0; color: #7f8c8d; font-size: 0.9em; text-transform: uppercase; }
        .card .value { font-size: 2em; font-weight: bold; color: #2c3e50; }
        .score-box { background: linear-gradient(135deg, #74b9ff, #0984e3); color: white; padding: 30px; border-radius: 10px; text-align: center; margin: 30px 0; }
        .score-box .score { font-size: 4em; font-weight: bold; }
        .score-box .status { font-size: 1.5em; margin-top: 10px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; }
        .limitations { background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 20px; margin-top: 30px; }
        .limitations h3 { color: #856404; margin-top: 0; }
        .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 0.9em; text-align: center; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }
        .stat-card { background: #f8f9fa; border-radius: 8px; padding: 20px; text-align: center; }
        .stat-card .value { font-size: 2em; font-weight: bold; color: #2c3e50; }
        .stat-card .label { color: #7f8c8d; font-size: 0.9em; }
        .captures-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .capture-card { background: white; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }
        .capture-header { background: #34495e; color: white; padding: 10px 15px; display: flex; justify-content: space-between; }
        .capture-num { font-weight: bold; }
        .capture-card img { width: 100%; height: auto; }
        .capture-stats { padding: 15px; display: flex; justify-content: space-around; background: #f8f9fa; }
        .weather-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }
        .weather-card { background: #e3f2fd; border-radius: 8px; padding: 15px; text-align: center; }
    """

def generate_html_report(
    bridge_config: BridgeConfig,
    vehicle_data: Dict,
    weather: Dict,
    scenario_result: Dict,
    baseline_metrics: Dict
) -> str:
    """Generate HTML report for download"""
    
    status_text, _ = get_status(scenario_result["combined_fatigue"])
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bridge Fatigue Assessment Report</title>
        <style>
            {get_report_css()}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🌉 Hybrid Digital Twin</h1>
            <p>Bridge Fatigue Assessment Report</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <h2>Bridge Information</h2>
        <div class="grid">
            <div class="card">
                <h3>Structure</h3>
                <div class="value">{bridge_config.name}</div>
            </div>
            <div class="card">
                <h3>Age</h3>
                <div class="value">{bridge_config.age_years} years</div>
            </div>
            <div class="card">
                <h3>Material</h3>
                <div class="value">{bridge_config.material}</div>
            </div>
            <div class="card">
                <h3>Main Span</h3>
                <div class="value">{bridge_config.main_span_m}m</div>
            </div>
        </div>
        
        <h2>Current Traffic Observation</h2>
        <div class="grid">
            <div class="card success">
                <h3>Vehicles on Bridge</h3>
                <div class="value">{vehicle_data.get('approaching', {}).get('total', 'N/A')}</div>
            </div>
            <div class="card">
                <h3>Current Load</h3>
                <div class="value">{vehicle_data.get('load_tons', 'N/A')} tons</div>
            </div>
            <div class="card">
                <h3>Traffic Density</h3>
                <div class="value">{vehicle_data.get('density', 'N/A')}</div>
            </div>
        </div>
        
        <h2>Weather Conditions</h2>
        <div class="grid">
            <div class="card">
                <h3>Temperature</h3>
                <div class="value">{weather.get('temperature', 'N/A')}°C</div>
            </div>
            <div class="card">
                <h3>Humidity</h3>
                <div class="value">{weather.get('humidity', 'N/A')}%</div>
            </div>
            <div class="card">
                <h3>Precipitation</h3>
                <div class="value">{weather.get('precipitation', 'N/A')} mm</div>
            </div>
            <div class="card warning">
                <h3>Freeze-Thaw (7 days)</h3>
                <div class="value">{weather.get('freeze_thaw_7day', 'N/A')} cycles</div>
            </div>
        </div>
        
        <div class="score-box" style="background: linear-gradient(135deg, {'#27ae60, #2ecc71' if scenario_result['combined_fatigue'] < 50 else '#e74c3c, #c0392b' if scenario_result['combined_fatigue'] > 70 else '#f39c12, #e67e22'});">
            <div class="score">{scenario_result['combined_fatigue']}/100</div>
            <div class="status">{status_text}</div>
        </div>
        
        <h2>Fatigue Breakdown</h2>
        <table>
            <tr>
                <th>Component</th>
                <th>Score</th>
                <th>Weight</th>
                <th>Contribution</th>
            </tr>
            <tr>
                <td>Traffic Stress</td>
                <td>{scenario_result['traffic_fatigue']}</td>
                <td>70%</td>
                <td>{scenario_result['traffic_fatigue'] * 0.7:.1f}</td>
            </tr>
            <tr>
                <td>Environmental Stress</td>
                <td>{scenario_result['environmental_stress']}</td>
                <td>30%</td>
                <td>{scenario_result['environmental_stress'] * 0.3:.1f}</td>
            </tr>
            <tr style="font-weight: bold;">
                <td>Combined</td>
                <td colspan="3">{scenario_result['combined_fatigue']}</td>
            </tr>
        </table>
        
        <div class="limitations">
            <h3>⚠️ Critical Limitations</h3>
            <p>This is a <strong>proof-of-concept demonstration</strong> with significant limitations:</p>
            <ul>
                <li>Material properties are estimated, not measured from actual bridge specifications</li>
                <li>Fatigue score is a proxy metric, not validated against structural sensors</li>
                <li>Environmental correlations based on engineering literature, not site-specific calibration</li>
                <li>Computer vision load estimates lack weigh-in-motion validation</li>
            </ul>
            <p>This framework demonstrates feasibility of low-cost monitoring, not operational readiness.</p>
        </div>
        
        <div class="footer">
            <p>Hybrid Digital Twin - Bridge Fatigue Monitoring System</p>
            <p>Case Study: Twin Bridges (Peace Bridge)</p>
            <p>For research and demonstration purposes only</p>
        </div>
    </body>
    </html>
    """
    
    return html


def generate_session_report(
    bridge_config: BridgeConfig,
    session_log: list,
    weather: Dict
) -> str:
    """Generate HTML report for monitoring session with images"""
    
    if not session_log:
        return "<html><body><p>No data captured</p></body></html>"
    
    # Calculate session stats
    total_vehicles = sum(entry["vehicle_data"]["approaching"]["total"] for entry in session_log)
    avg_vehicles = total_vehicles / len(session_log)
    avg_load = sum(entry["vehicle_data"]["load_tons"] for entry in session_log) / len(session_log)
    max_load = max(entry["vehicle_data"]["load_tons"] for entry in session_log)
    avg_beta = sum(float(e.get("beta_primary") or 0.0) for e in session_log) / len(session_log)


    
    start_time = session_log[0]["timestamp"]
    end_time = session_log[-1]["timestamp"]
    
    # Build capture rows
    capture_rows = ""
    for i, entry in enumerate(session_log):
        vd = entry["vehicle_data"]
        approaching = vd.get("approaching", {})
        capture_rows += f"""
        <div class="capture-card">
            <div class="capture-header">
                <span class="capture-num">#{i+1}</span>
                <span class="capture-time">{entry['timestamp'].strftime('%H:%M:%S')}</span>
            </div>
            <img src="data:image/jpeg;base64,{entry['image_b64']}" alt="Capture {i+1}" />
            <div class="capture-stats">
                <div><strong>{approaching.get('total', 0)}</strong> vehicles</div>
                <div><strong>{vd.get('load_tons', 0)}</strong> tons</div>
                <div>🚗 {approaching.get('car', 0)} 🚛 {approaching.get('truck', 0)} 🚌 {approaching.get('bus', 0)}</div>
            </div>
        </div>
        """
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bridge Monitoring Session Report</title>
        <style>
            {get_report_css()}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📹 Bridge Monitoring Session Report</h1>
            <p><strong>{bridge_config.name}</strong></p>
            <p>Session: {start_time.strftime('%Y-%m-%d %H:%M:%S')} - {end_time.strftime('%H:%M:%S')}</p>
            <p>{len(session_log)} captures recorded</p>
        </div>
        
        <h2>📊 Session Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{len(session_log)}</div>
                <div class="label">Captures</div>
            </div>
            <div class="stat-card">
                <div class="value">{avg_vehicles:.1f}</div>
                <div class="label">Avg Vehicles</div>
            </div>
            <div class="stat-card">
                <div class="value">{avg_load:.1f}</div>
                <div class="label">Avg Load (tons)</div>
            </div>
            <div class="stat-card">
                <div class="value">{max_load:.1f}</div>
                <div class="label">Max Load (tons)</div>
            </div>
            <div class="stat-card">
                <div class="value">{avg_beta:.2f}</div>
                <div class="label">Avg Reliability β</div>
            </div>
        </div>
        
        <h2>🌤️ Weather Conditions</h2>
        <div class="weather-grid">
            <div class="weather-card">
                <div style="font-size: 1.5em;">{weather.get('temperature', 'N/A')}°C</div>
                <div>Temperature</div>
            </div>
            <div class="weather-card">
                <div style="font-size: 1.5em;">{weather.get('humidity', 'N/A')}%</div>
                <div>Humidity</div>
            </div>
            <div class="weather-card">
                <div style="font-size: 1.5em;">{weather.get('precipitation', 'N/A')} mm</div>
                <div>Precipitation</div>
            </div>
            <div class="weather-card">
                <div style="font-size: 1.5em;">{weather.get('freeze_thaw_7day', 'N/A')}</div>
                <div>F/T Cycles (7d)</div>
            </div>
        </div>
        
        <h2>📷 Captured Frames</h2>
        <div class="captures-grid">
            {capture_rows}
        </div>
        
        <div class="limitations">
            <h3>⚠️ Limitations</h3>
            <p>This monitoring session data is for <strong>research demonstration purposes only</strong>. 
            Vehicle detection may have errors. Load estimates are approximations based on average vehicle weights.
            This system has not been validated against physical sensors.</p>
        </div>
        
        <div class="footer">
            <p>Hybrid Digital Twin - Bridge Fatigue Monitoring System</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </body>
    </html>
    """
    
    return html


def generate_nysdot_comparison_report(
    bridge_config: BridgeConfig,
    session_log: list,
    nysdot_data: Optional[pd.DataFrame] = None
) -> str:
    """
    Generate HTML report template for comparing detection results with NYSDOT data.
    If nysdot_data is None, generates a template with placeholder for manual entry.
    """
    
    if not session_log:
        return "<html><body><p>No session data</p></body></html>"
    
    # Aggregate session data by hour
    session_df = pd.DataFrame([
        {
            "timestamp": entry["timestamp"],
            "hour": entry["timestamp"].strftime("%H:00"),
            "vehicles": entry["vehicle_data"]["approaching"]["total"],
            "cars": entry["vehicle_data"]["approaching"]["car"],
            "trucks": entry["vehicle_data"]["approaching"]["truck"],
            "load_tons": entry["vehicle_data"]["load_tons"]
        }
        for entry in session_log
    ])
    
    # Aggregate by hour
    hourly = session_df.groupby("hour").agg({
        "vehicles": "sum",
        "cars": "sum",
        "trucks": "sum",
        "load_tons": "mean"
    }).reset_index()
    
    total_detected = session_df["vehicles"].sum()
    session_duration = (session_log[-1]["timestamp"] - session_log[0]["timestamp"]).total_seconds() / 3600
    
    # Build hourly comparison rows
    comparison_rows = ""
    for _, row in hourly.iterrows():
        comparison_rows += f"""
        <tr>
            <td>{row['hour']}</td>
            <td>{int(row['vehicles'])}</td>
            <td><input type="number" class="nysdot-input" placeholder="Enter NYSDOT count"></td>
            <td class="diff-cell">-</td>
            <td class="error-cell">-</td>
        </tr>
        """
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>NYSDOT Validation Report - {bridge_config.name}</title>
        <style>
            {get_report_css()}
            /* Additional styles for this report */
            .info-box {{ background: #e3f2fd; border-radius: 8px; padding: 20px; margin: 20px 0; }}
            tr:hover {{ background: #f5f5f5; }}
            .nysdot-input {{ width: 100px; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }}
            .diff-cell, .error-cell {{ font-weight: bold; }}
            .positive {{ color: #27ae60; }}
            .negative {{ color: #e74c3c; }}
            .summary-box {{ background: #f8f9fa; border: 2px solid #3498db; border-radius: 8px; padding: 20px; margin: 30px 0; }}
            .summary-box h3 {{ margin-top: 0; color: #3498db; }}
            .instructions {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 20px; margin: 20px 0; }}
            .instructions h3 {{ color: #856404; margin-top: 0; }}
            .instructions ol {{ margin-bottom: 0; }}
            button {{ background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 1em; }}
            button:hover {{ background: #2980b9; }}
            .results {{ display: none; margin-top: 20px; padding: 20px; background: #d4edda; border-radius: 8px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 NYSDOT Validation Report</h1>
            <p>Comparing Camera Detection vs Official Traffic Counts</p>
            <p><strong>{bridge_config.name}</strong> | {bridge_config.location}</p>
        </div>
        
        <div class="instructions">
            <h3>📝 Instructions</h3>
            <ol>
                <li>Go to <a href="https://gisportalny.dot.ny.gov/portalny/apps/webappviewer/index.html?id=28537cbc8b5941e19cf8e959b16797b4" target="_blank">NYSDOT Traffic Data Viewer</a></li>
                <li>Navigate to Twin Bridges / I-87 near Cohoes, NY</li>
                <li>Click on the nearest Continuous Count station</li>
                <li>Download hourly data for <strong>{session_log[0]["timestamp"].strftime('%Y-%m-%d')}</strong></li>
                <li>Enter the hourly counts in the table below</li>
                <li>Click "Calculate Accuracy" to see comparison</li>
            </ol>
        </div>
        
        <h2>📈 Detection Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{total_detected}</div>
                <div class="label">Total Vehicles Detected</div>
            </div>
            <div class="stat-card">
                <div class="value">{session_duration:.1f} hrs</div>
                <div class="label">Session Duration</div>
            </div>
            <div class="stat-card">
                <div class="value">{total_detected / max(session_duration, 0.1):.0f}</div>
                <div class="label">Vehicles/Hour (Avg)</div>
            </div>
            <div class="stat-card">
                <div class="value">{len(session_log)}</div>
                <div class="label">Capture Points</div>
            </div>
        </div>
        
        <h2>⚖️ Hourly Comparison</h2>
        <table id="comparison-table">
            <thead>
                <tr>
                    <th>Hour</th>
                    <th>Detected (Camera)</th>
                    <th>NYSDOT Official</th>
                    <th>Difference</th>
                    <th>Error %</th>
                </tr>
            </thead>
            <tbody>
                {comparison_rows}
            </tbody>
            <tfoot>
                <tr style="background: #f8f9fa; font-weight: bold;">
                    <td>TOTAL</td>
                    <td id="total-detected">{total_detected}</td>
                    <td id="total-nysdot">-</td>
                    <td id="total-diff">-</td>
                    <td id="total-error">-</td>
                </tr>
            </tfoot>
        </table>
        
        <button onclick="calculateAccuracy()">📊 Calculate Accuracy</button>
        
        <div id="results" class="results">
            <h3>✅ Validation Results</h3>
            <p id="result-text"></p>
        </div>
        
        <div class="summary-box">
            <h3>📋 For Accuracy Comparison Report</h3>
            <p>Enter the NYSDOT data, and the system will generate a report:</p>
            <blockquote id="paper-quote" style="font-style: italic; background: white; padding: 15px; border-left: 4px solid #3498db;">
                "Vehicle detection accuracy was validated against NYSDOT continuous count station data for I-87. 
                Over a [DURATION]-hour monitoring period on [DATE], the system detected [DETECTED] vehicles 
                compared to the official count of [OFFICIAL], representing a [ERROR]% [OVER/UNDER]count."
            </blockquote>
        </div>
        
        <div class="footer">
            <p>Hybrid Digital Twin - Bridge Fatigue Monitoring System</p>
            <p>Session Date: {session_log[0]["timestamp"].strftime('%Y-%m-%d')}</p>
            <p>Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <script>
            function calculateAccuracy() {{
                const inputs = document.querySelectorAll('.nysdot-input');
                const rows = document.querySelectorAll('#comparison-table tbody tr');
                let totalDetected = 0;
                let totalNysdot = 0;
                
                rows.forEach((row, index) => {{
                    const detected = parseInt(row.cells[1].textContent);
                    const nysdotValue = parseInt(inputs[index].value) || 0;
                    
                    if (nysdotValue > 0) {{
                        const diff = detected - nysdotValue;
                        const errorPct = ((diff / nysdotValue) * 100).toFixed(1);
                        
                        row.cells[3].textContent = (diff > 0 ? '+' : '') + diff;
                        row.cells[3].className = 'diff-cell ' + (diff >= 0 ? 'positive' : 'negative');
                        
                        row.cells[4].textContent = errorPct + '%';
                        row.cells[4].className = 'error-cell ' + (Math.abs(errorPct) < 10 ? 'positive' : 'negative');
                        
                        totalDetected += detected;
                        totalNysdot += nysdotValue;
                    }}
                }});
                
                if (totalNysdot > 0) {{
                    const totalDiff = totalDetected - totalNysdot;
                    const totalErrorPct = ((totalDiff / totalNysdot) * 100).toFixed(1);
                    
                    document.getElementById('total-nysdot').textContent = totalNysdot;
                    document.getElementById('total-diff').textContent = (totalDiff > 0 ? '+' : '') + totalDiff;
                    document.getElementById('total-error').textContent = totalErrorPct + '%';
                    
                    const resultDiv = document.getElementById('results');
                    resultDiv.style.display = 'block';
                    
                    const accuracy = (100 - Math.abs(totalErrorPct)).toFixed(1);
                    const overUnder = totalDiff > 0 ? 'overcount' : 'undercount';
                    
                    document.getElementById('result-text').innerHTML = 
                        '<strong>Overall Accuracy: ' + accuracy + '%</strong><br>' +
                        'Total Detected: ' + totalDetected + '<br>' +
                        'Total NYSDOT: ' + totalNysdot + '<br>' +
                        'Difference: ' + (totalDiff > 0 ? '+' : '') + totalDiff + ' (' + Math.abs(totalErrorPct) + '% ' + overUnder + ')';
                    
                    // Update paper quote
                    document.getElementById('paper-quote').innerHTML = 
                        '"Vehicle detection accuracy was validated against NYSDOT continuous count station data for I-87. ' +
                        'Over a <strong>{session_duration:.1f}</strong>-hour monitoring period on <strong>{session_log[0]["timestamp"].strftime('%Y-%m-%d')}</strong>, ' +
                        'the system detected <strong>' + totalDetected + '</strong> vehicles compared to the official count of ' +
                        '<strong>' + totalNysdot + '</strong>, representing a <strong>' + Math.abs(totalErrorPct) + '%</strong> ' + overUnder + '."';
                }}
            }}
        </script>
    </body>
    </html>
    """
    
    return html


def export_results_bundle(
    out_dir: str,
    session_log: list,
    mc_df: Optional[pd.DataFrame] = None,
    validation: Optional[Dict] = None,
    rf_metrics: Optional[Dict] = None
) -> None:
    import os, json
    os.makedirs(out_dir, exist_ok=True)

    # raw session
    pd.DataFrame(session_log).to_pickle(os.path.join(out_dir, "session_log.pkl"))

    # monte carlo
    if mc_df is not None:
        mc_df.to_csv(os.path.join(out_dir, "monte_carlo.csv"), index=False)

    # validation
    if validation is not None:
        if "hourly_table" in validation and isinstance(validation["hourly_table"], pd.DataFrame):
            validation["hourly_table"].to_csv(os.path.join(out_dir, "validation_hourly.csv"), index=False)
            validation = {k: v for k, v in validation.items() if k != "hourly_table"}
        with open(os.path.join(out_dir, "validation_metrics.json"), "w") as f:
            json.dump(validation, f, indent=2)

    # rf
    if rf_metrics is not None:
        rf_out = dict(rf_metrics)
        if "folds_df" in rf_out and isinstance(rf_out["folds_df"], pd.DataFrame):
            rf_out["folds_df"].to_csv(os.path.join(out_dir, "rf_cv_folds.csv"), index=False)
            rf_out.pop("folds_df", None)
        with open(os.path.join(out_dir, "rf_metrics.json"), "w") as f:
            json.dump(rf_out, f, indent=2)

# =============================================================================
# 9) MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="Hybrid Digital Twin - Bridge Monitoring",
        page_icon="🌉",
        layout="wide"
    )
    
    # Initialize
    bridge_config = BridgeConfig()
    
    # Check dependencies
    if not YOLO_AVAILABLE:
        st.error("YOLO not available. Install: `pip install ultralytics`")
        st.stop()

    # =========================================================================
    # INITIALIZE SESSION STATE
    # =========================================================================
    if "baseline_calculated" not in st.session_state:
        st.session_state.baseline_calculated = False
        st.session_state.baseline_traffic_fatigue = 50
        st.session_state.baseline_env_stress = 30
        st.session_state.mc_data = None
        st.session_state.rf_model = None
        st.session_state.rf_metrics = {}
        st.session_state.vehicle_data = {}
        st.session_state.weather = {}
        # Initialize weights in session state
        st.session_state.vehicle_weights = VEHICLE_WEIGHTS.copy()
    
    if "monitoring_active" not in st.session_state:
        st.session_state.monitoring_active = False
        st.session_state.monitoring_start_time = None
        st.session_state.session_log = []  # List of {timestamp, vehicle_data, image_b64}
    if "live_capture_log" not in st.session_state:
        st.session_state.live_capture_log = []
    if "auto_monitoring_enabled" not in st.session_state:
        st.session_state.auto_monitoring_enabled = False
    if "auto_monitoring_started" not in st.session_state:
        st.session_state.auto_monitoring_started = False
    if "validation_active" not in st.session_state:
        st.session_state.validation_active = False
    if "validation_duration_min" not in st.session_state:
        st.session_state.validation_duration_min = 60
    if "validation_interval_sec" not in st.session_state:
        st.session_state.validation_interval_sec = 30
    if "k_mpa_per_ton" not in st.session_state:
        st.session_state.k_mpa_per_ton = 0.6
    if "stress_ref_mpa" not in st.session_state:
        st.session_state.stress_ref_mpa = 80.0
    if "prev_detections" not in st.session_state:
        st.session_state.prev_detections = None
    if "prev_detections_time" not in st.session_state:
        st.session_state.prev_detections_time = None
    if "latest_frame_rgb" not in st.session_state:
        st.session_state.latest_frame_rgb = None
    if "latest_frame_time" not in st.session_state:
        st.session_state.latest_frame_time = None
    if "last_capture_time" not in st.session_state:
        st.session_state.last_capture_time = None
    if "speed_ema_mps" not in st.session_state:
        st.session_state.speed_ema_mps = None
    if "speed_mode" not in st.session_state:
        st.session_state.speed_mode = "depth"
    if "pixel_m_per_px" not in st.session_state:
        st.session_state.pixel_m_per_px = 0.05
    if "pixel_axis" not in st.session_state:
        st.session_state.pixel_axis = "x"
    if "focal_length_px" not in st.session_state:
        st.session_state.focal_length_px = None
    if "experiment_runs" not in st.session_state:
        st.session_state.experiment_runs = []
        st.session_state.experiment_run_active = False
        st.session_state.experiment_run_id = 0
        st.session_state.exp_settings = {
            "interval_sec": 1.0,
            "duration_min": 10,
            "confidence": 0.15,
            "use_roi": True,
            "roi": None,
            "traffic_weight": 0.70,
            "environment_weight": 0.30,
            "use_live_weather": True,
            "enabled": False
        }
    if "experiment_mode" not in st.session_state:
        st.session_state.experiment_mode = False
        st.session_state.exp_density = 0.05
        st.session_state.exp_vmax = 80
        st.session_state.exp_jam = False
        st.session_state.run_single_experiment = False
        st.session_state.experiment_log = []
    if "log_detections_for_validation" not in st.session_state:
        st.session_state.log_detections_for_validation = False
    
    if "yolo_model" not in st.session_state:
        with st.spinner("Loading YOLO model..."):
            st.session_state.yolo_model = YOLO("yolov8n.pt")

  
    # =========================================================================
    # HEADER
    # =========================================================================
    st.title("🌉 Hybrid Digital Twin - Bridge Fatigue Monitoring")
    st.markdown(
        f"**{bridge_config.name}** | "
        f"Built {bridge_config.year_built} | "
        f"Age: {bridge_config.age_years} years | "
        f"*Proof-of-concept: Low-cost monitoring using existing cameras*"
    )
    
    # =========================================================================
    # STAGE 1: VIDEO SOURCE & MONITORING CONTROLS
    # =========================================================================
    st.markdown("### 📹 Video Source")
    
    col_source, col_monitor = st.columns([2, 1])
    
    with col_source:
        selected_camera = st.selectbox("Camera", list(CAMERAS.keys()), label_visibility="collapsed")
        camera_config = CAMERAS[selected_camera]
    
    with col_monitor:
        col_start, col_stop = st.columns(2)
        with col_start:
            start_btn = st.button("▶️ START", type="primary", 
                                  disabled=st.session_state.monitoring_active,
                                  use_container_width=True)
        with col_stop:
            stop_btn = st.button("⏹️ STOP", 
                                 disabled=not st.session_state.monitoring_active,
                                 use_container_width=True)
        
        if start_btn:
            st.session_state.monitoring_active = True
            st.session_state.monitoring_start_time = datetime.now()
            st.session_state.session_log = []
            st.rerun()
        if stop_btn:
            st.session_state.monitoring_active = False
            st.session_state.validation_active = False
            st.rerun()
    
    # Capture settings inline
    col_interval, col_duration, col_auto = st.columns(3)
    with col_interval:
        capture_interval = st.select_slider(
            "Capture every",
            options=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 5.0, 7.5, 10.0],
            value=2.0,
            format_func=lambda x: f"{x}s"
        )
    with col_duration:
        monitor_duration = st.select_slider(
            "Duration",
            options=[10, 15, 20],
            value=15,
            format_func=lambda x: f"{x} min"
        )
    with col_auto:
        auto_monitoring_enabled = st.toggle(
            "Auto-capture",
            value=st.session_state.auto_monitoring_enabled,
            help="Automatically run a monitoring session"
        )
        st.session_state.auto_monitoring_enabled = auto_monitoring_enabled

    # =========================================================================
    # STAGE 2: DETECTION SETTINGS
    # =========================================================================
    with st.expander("🔍 Detection Settings", expanded=False):
        col_det1, col_det2, col_det3 = st.columns(3)
        
        with col_det1:
            confidence = st.slider("Confidence", 0.05, 0.50, 0.15, 0.05)
        with col_det2:
            lane_divider = st.slider("Lane Divider", 0.3, 0.7, 0.43, 0.01)
        with col_det3:
            speed_mode = st.selectbox(
                "Speed Mode",
                ["depth", "pixel"],
                index=0 if st.session_state.speed_mode == "depth" else 1,
                help="Depth: bbox width | Pixel: center displacement"
            )
            st.session_state.speed_mode = speed_mode
        
        if speed_mode == "pixel":
            col_axis, col_scale = st.columns(2)
            with col_axis:
                st.session_state.pixel_axis = st.selectbox(
                    "Pixel axis", ["x", "y", "diag"], 
                    index=["x", "y", "diag"].index(st.session_state.pixel_axis)
                )
            with col_scale:
                st.session_state.pixel_m_per_px = st.slider(
                    "Meters/pixel", 0.005, 0.20, 
                    float(st.session_state.pixel_m_per_px), 0.005
                )
        
        # ROI Settings
        st.markdown("**ROI (Region of Interest)**")
        use_roi = st.checkbox("Enable ROI Box", value=True,
                              help="Only count vehicles within the defined box")
        
        if use_roi:
            col_roi1, col_roi2, col_roi3, col_roi4 = st.columns(4)
            with col_roi1:
                roi_x1 = st.slider("Left %", 0, 50, 20, key="roi_x1")
            with col_roi2:
                roi_y1 = st.slider("Top %", 0, 50, 35, key="roi_y1")
            with col_roi3:
                roi_x2 = st.slider("Right %", 50, 100, 80, key="roi_x2")
            with col_roi4:
                roi_y2 = st.slider("Bottom %", 50, 100, 75, key="roi_y2")
            
            roi_override = {
                "x1_pct": roi_x1 / 100,
                "y1_pct": roi_y1 / 100,
                "x2_pct": roi_x2 / 100,
                "y2_pct": roi_y2 / 100
            }
        else:
            roi_override = None
        
        st.session_state.use_roi = use_roi
        st.session_state.roi_override = roi_override

    # =========================================================================
    # STAGE 3: ADVANCED SETTINGS (collapsed by default)
    # =========================================================================
    with st.expander("⚙️ Advanced Settings", expanded=False):
        adv_tab1, adv_tab2, adv_tab3, adv_tab4 = st.tabs([
            "🚗 Vehicle Weights", "🧪 Experiments", "📊 Simulation", "✅ Validation"
        ])
        
        with adv_tab1:
            col_w1, col_w2 = st.columns(2)
            with col_w1:
                car_weight = st.number_input("Car (lbs)", 2000, 6000, 4000, 500)
                truck_weight = st.number_input("Truck (lbs)", 15000, 80000, 35000, 5000)
            with col_w2:
                bus_weight = st.number_input("Bus (lbs)", 15000, 40000, 25000, 2500)
                motorcycle_weight = st.number_input("Motorcycle (lbs)", 200, 2000, 500, 50)
            
            st.session_state.vehicle_weights['car'] = car_weight
            st.session_state.vehicle_weights['truck'] = truck_weight
            st.session_state.vehicle_weights['bus'] = bus_weight
            st.session_state.vehicle_weights['motorcycle'] = motorcycle_weight
        
        with adv_tab2:
            experiment_mode = st.toggle("Enable Experiment Mode", value=st.session_state.experiment_mode)
            st.session_state.experiment_mode = experiment_mode
            
            if experiment_mode:
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    exp_density = st.slider("Density (veh/m)", 0.01, 0.15, st.session_state.exp_density, 0.005)
                    exp_jam = st.checkbox("Inject Jam Event", value=st.session_state.exp_jam)
                with col_exp2:
                    exp_vmax = st.slider("Free-flow Speed (km/h)", 40, 120, st.session_state.exp_vmax, 5)
                    if st.button("▶️ Run Experiment"):
                        st.session_state.run_single_experiment = True
                
                st.session_state.exp_density = exp_density
                st.session_state.exp_vmax = exp_vmax
                st.session_state.exp_jam = exp_jam
            
            st.markdown("---")
            exp_on = st.toggle(
                "Enable Experiment Settings",
                value=st.session_state.get("exp_settings", {}).get("enabled", False),
                help="Apply custom settings for experiment runs"
            )
            
            if exp_on:
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    exp_interval = st.slider("Interval (sec)", 0.5, 10.0,
                        float(st.session_state.get("exp_settings", {}).get("interval_sec", 1.0)), 0.5)
                    exp_conf = st.slider("Confidence", 0.05, 0.50,
                        float(st.session_state.get("exp_settings", {}).get("confidence", confidence)), 0.05)
                    traffic_w = st.slider("Traffic weight", 0.0, 1.0,
                        float(st.session_state.get("exp_settings", {}).get("traffic_weight", 0.70)), 0.05)
                with col_e2:
                    exp_duration = st.slider("Duration (min)", 10, 20,
                        int(st.session_state.get("exp_settings", {}).get("duration_min", 10)), 1)
                    exp_use_roi = st.checkbox("Use ROI", value=bool(st.session_state.get("use_roi", True)))
                    use_live_weather = st.checkbox("Use live weather",
                        value=bool(st.session_state.get("exp_settings", {}).get("use_live_weather", True)))
                
                env_w = round(1.0 - traffic_w, 2)
                roi_snap = st.session_state.get("roi_override", None)
                
                if st.button("🚀 Apply & Run Experiment", type="primary"):
                    st.session_state.exp_settings = {
                        "enabled": exp_on,
                        "interval_sec": float(exp_interval),
                        "duration_min": int(exp_duration),
                        "confidence": float(exp_conf),
                        "use_roi": bool(exp_use_roi),
                        "roi": roi_snap,
                        "traffic_weight": float(traffic_w),
                        "environment_weight": float(env_w),
                        "use_live_weather": bool(use_live_weather),
                    }
                    st.session_state.use_roi = bool(exp_use_roi)
                    st.session_state.roi_override = roi_snap
                    st.session_state.experiment_run_id = int(st.session_state.get("experiment_run_id", 0)) + 1
                    st.session_state.experiment_run_active = True
                    st.session_state.monitoring_active = True
                    st.session_state.monitoring_start_time = datetime.now()
                    st.session_state.session_log = []
                    st.session_state.auto_monitoring_enabled = False
                    st.session_state.validation_active = False
                    st.rerun()
            
            if st.session_state.get("experiment_runs"):
                st.markdown("**Experiment History**")
                st.dataframe(
                    pd.DataFrame(st.session_state.experiment_runs)[
                        ["run_id", "start", "end", "captures", "avg_load_tons", "max_beta", "alerts"]
                    ],
                    use_container_width=True,
                    hide_index=True
                )
                if st.button("Clear History"):
                    st.session_state.experiment_runs = []
        
        with adv_tab3:
            col_sim1, col_sim2 = st.columns(2)
            with col_sim1:
                mc_runs = st.slider("Monte Carlo Runs", 50, 300, 100, 50)
                jam_probability = st.slider("Jam Probability", 0.0, 1.0, 0.3, 0.1)
            with col_sim2:
                st.session_state.k_mpa_per_ton = st.slider(
                    "Stress proxy k (MPa/ton)", 0.1, 2.0,
                    float(st.session_state.k_mpa_per_ton), 0.1,
                    help="Convert load (tons) to stress proxy"
                )
            
            st.session_state.log_detections_for_validation = st.checkbox(
                "Log detections for validation (memory heavy)",
                value=st.session_state.log_detections_for_validation
            )
        
        with adv_tab4:
            st.markdown("**Validation Run Settings**")
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                validation_interval = st.select_slider(
                    "Capture every",
                    options=[10, 15, 30, 45, 60, 120, 300],
                    value=st.session_state.validation_interval_sec,
                    format_func=lambda x: f"{x}s" if x < 60 else f"{x//60}m"
                )
            with col_v2:
                validation_duration = st.select_slider(
                    "Duration",
                    options=[10, 15, 30, 45, 60, 90, 120],
                    value=st.session_state.validation_duration_min,
                    format_func=lambda x: f"{x} min"
                )
            
            if st.button("▶️ Start Validation Run", type="primary"):
                st.session_state.validation_interval_sec = validation_interval
                st.session_state.validation_duration_min = validation_duration
                st.session_state.monitoring_active = True
                st.session_state.monitoring_start_time = datetime.now()
                st.session_state.session_log = []
                st.session_state.validation_active = True
                st.rerun()

    st.caption("*Proof-of-Concept: Fatigue scores are proxy metrics, not validated against real sensors.*")


    # =========================================================================
    # LOAD MODELS
    # =========================================================================
    if "yolo_model" not in st.session_state:
        with st.spinner("Loading YOLO model..."):
            st.session_state.yolo_model = YOLO("yolov8n.pt")
    
    if "baseline_calculated" not in st.session_state:
        st.session_state.baseline_calculated = False
        st.session_state.baseline_traffic_fatigue = 50
        st.session_state.baseline_env_stress = 30
        st.session_state.mc_data = None
        st.session_state.rf_model = None
        st.session_state.rf_metrics = {}
        st.session_state.vehicle_data = {}
        st.session_state.weather = {}
        # Initialize weights in session state
        st.session_state.vehicle_weights = VEHICLE_WEIGHTS.copy()
    
    # Monitoring session state
    if "monitoring_active" not in st.session_state:
        st.session_state.monitoring_active = False
        st.session_state.monitoring_start_time = None
        st.session_state.session_log = []  # List of {timestamp, vehicle_data, image_b64}
    
    # =========================================================================
    # LIVE FEED + CURRENT CONDITIONS
    # =========================================================================
    st.markdown("---")
    
    col_video, col_conditions = st.columns([2, 1])
    
    with col_video:
        col_stream, col_live = st.columns(2)

        with col_stream:
            st.subheader("Live Stream (YouTube)")
            st.video(camera_config["url"])
            st.caption("Embedded stream is separate from detection frames.")

        with col_live:
            st.subheader("Live Camera Feed")
            video_placeholder = st.empty()
            status_placeholder = st.empty()
            
            with st.expander("Live feed controls", expanded=False):
                if st.button("Refresh Frame"):
                    pass  # Just triggers rerun
                auto_refresh_live = st.toggle("Auto-refresh live feed", value=True)
                live_refresh_sec = st.slider("Live refresh (seconds)", 0.05, 10.0, 2.0, 0.5)
                st.caption(f"Auto-refresh interval: {live_refresh_sec:.1f}s")
                if st.session_state.live_capture_log:
                    live_csv = pd.DataFrame(st.session_state.live_capture_log).to_csv(index=False)
                    st.download_button(
                        "Download Live Feed Log (CSV)",
                        live_csv,
                        f"live_feed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        width='stretch'
                    )
            
            # Capture frame
            status_placeholder.info("Connecting to camera...")
            if st.session_state.monitoring_active and st.session_state.latest_frame_rgb is not None:
                video_placeholder.image(st.session_state.latest_frame_rgb, width='stretch')
                ts = st.session_state.latest_frame_time or datetime.now()
                status_placeholder.success(f"Live (session) | {ts.strftime('%H:%M:%S')}")
            elif st.session_state.monitoring_active and st.session_state.latest_frame_rgb is None:
                status_placeholder.info("Session running,waiting for first scheduled capture to populate the live frame.")
            else:
                frame = capture_frame(camera_config["url"])
                
                if frame is not None:
                    now = datetime.now()
                    exp_cfg = st.session_state.get("exp_settings", {}) or {}
                    exp_active = bool(st.session_state.get("experiment_run_active", False))
                    eff_conf = float(exp_cfg.get("confidence", confidence)) if exp_active else confidence
                    eff_use_roi = bool(exp_cfg.get("use_roi", st.session_state.get("use_roi", True)))
                    eff_roi_override = exp_cfg.get("roi", st.session_state.get("roi_override", None))
                    st.session_state.last_capture_time = now
                    vehicle_data, annotated_frame, detections = detect_vehicles(
                        frame=frame,
                        model=st.session_state.yolo_model,
                        camera_config=camera_config,
                        camera_name=selected_camera,
                        lane_divider=lane_divider,
                        confidence=eff_conf,
                        bridge_config=bridge_config,
                        use_roi=eff_use_roi,
                        roi_override=eff_roi_override,
                        weights=st.session_state.get("vehicle_weights", VEHICLE_WEIGHTS),
                        focal_length_px=st.session_state.get("focal_length_px"),
                    )
                    k = float(st.session_state.get("k_mpa_per_ton", 0.6))
                    stress_mpa = load_tons_to_stress_mpa(
                        vehicle_data.get("load_tons", 0.0),
                        k_mpa_per_ton=k
                    )
                    vehicle_data["stress_mpa_proxy"] = round(stress_mpa, 2)
                    prev_detections = st.session_state.get("prev_detections")
                    prev_time = st.session_state.get("prev_detections_time")
                    if prev_detections is not None and prev_time is not None:
                        dt_s = (now - prev_time).total_seconds()
                        if st.session_state.get("speed_mode") == "pixel":
                            avg_speed_mps = estimate_avg_speed_mps_pixel(
                                detections,
                                prev_detections,
                                dt_s,
                                meters_per_pixel=float(st.session_state.get("pixel_m_per_px", 0.05)),
                                axis=str(st.session_state.get("pixel_axis", "x")),
                                deadband_px=3.0,
                                dt_min_s=0.1,
                                dt_max_s=1.0,
                            )
                        else:
                            avg_speed_mps = estimate_avg_speed_mps(
                                detections,
                                prev_detections,
                                dt_s,
                                deadband_m=0.5,
                                dt_min_s=0.1,
                                dt_max_s=1.0,
                            )
                        if avg_speed_mps is not None:
                            ema_alpha = 0.3
                            prev_ema = st.session_state.get("speed_ema_mps")
                            if prev_ema is None:
                                st.session_state.speed_ema_mps = avg_speed_mps
                            else:
                                st.session_state.speed_ema_mps = (
                                    ema_alpha * avg_speed_mps + (1.0 - ema_alpha) * prev_ema
                                )
                            avg_speed_mps = st.session_state.speed_ema_mps
                        if avg_speed_mps is not None:
                            vehicle_data["avg_speed_mps"] = round(avg_speed_mps, 2)
                            vehicle_data["avg_speed_kph"] = round(avg_speed_mps * 3.6, 1)
                    st.session_state.vehicle_data = vehicle_data
                    st.session_state.latest_detections = detections
                    st.session_state.prev_detections = detections
                    st.session_state.prev_detections_time = now
                    st.session_state.live_capture_log.append({
                        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                        "approaching_total": vehicle_data.get("approaching", {}).get("total", 0),
                        "leaving_total": vehicle_data.get("leaving", {}).get("total", 0),
                        "load_tons": vehicle_data.get("load_tons", 0.0),
                        "density": vehicle_data.get("density", 0.0),
                        "avg_speed_kph": vehicle_data.get("avg_speed_kph", None),
                        "stress_mpa_proxy": vehicle_data.get("stress_mpa_proxy", 0.0),
                    })
                    if len(st.session_state.live_capture_log) > 500:
                        st.session_state.live_capture_log = st.session_state.live_capture_log[-500:]
                    
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, width='stretch')
                    status_placeholder.success(f"Live | {datetime.now().strftime('%H:%M:%S')}")
                else:
                    video_placeholder.warning(" Camera unavailable - using demo mode")
                    # Demo data
                    st.session_state.vehicle_data = {
                        "approaching": {"car": 5, "truck": 2, "bus": 1, "motorcycle": 0, "total": 8},
                        "load_tons": 42.5,
                        "density": 0.034
                    }

    with col_conditions:
        st.subheader("📊 Current Conditions")
        
        # Traffic
        vd = st.session_state.vehicle_data
        approaching = vd.get("approaching", {})
        leaving = vd.get("leaving", {})
        
        col_in, col_out = st.columns(2)
        with col_in:
            st.metric("⬅️ Incoming", approaching.get("total", 0))
            st.metric("Cars", approaching.get("car", 0))
            st.metric("Trucks", approaching.get("truck", 0))
            st.metric("Buses", approaching.get("bus", 0))
        with col_out:
            st.metric("➡️ Outgoing", leaving.get("total", 0))
            st.metric("Cars", leaving.get("car", 0))
            st.metric("Trucks", leaving.get("truck", 0))
            st.metric("Buses", leaving.get("bus", 0))
        
        st.metric("Load", f"{vd.get('load_tons', 0)} tons")
        if vd.get("avg_speed_kph") is not None:
            st.metric("Avg Speed", f"{vd.get('avg_speed_kph')} km/h")
    
        with st.expander("Weather", expanded=False):
            weather = fetch_weather(bridge_config.latitude, bridge_config.longitude)
            st.session_state.weather = weather
            
            col_w1, col_w2 = st.columns(2)
            with col_w1:
                st.metric("Temp", f"{weather['temperature']}°C")
                st.metric("Humidity", f"{weather['humidity']}%")
            with col_w2:
                st.metric("Precip", f"{weather['precipitation']} mm")
                st.metric("F/T Cycles", weather['freeze_thaw_7day'])

        # Detection Distance & ROI Analysis
        if vd.get("detection_stats"):
            stats = vd["detection_stats"]
            
            with st.expander("📏 Detection Distance & Coverage", expanded=False):
                col_d1, col_d2, col_d3, col_d4 = st.columns(4)
                with col_d1:
                    st.metric("Total Detected", stats.get("total_detected", 0))
                with col_d2:
                    st.metric("In ROI", stats.get("in_roi", 0))
                with col_d3:
                    st.metric("Outside ROI", stats.get("outside_roi", 0))
                with col_d4:
                    in_roi = stats.get("in_roi", 0)
                    total = stats.get("total_detected", 1)
                    roi_pct = (in_roi / total * 100) if total > 0 else 0
                    st.metric("ROI %", f"{roi_pct:.0f}%")
                
                col_e1, col_e2, col_e3 = st.columns(3)
                with col_e1:
                    st.metric("Nearest", f"{stats.get('min_distance_m', 0)} m")
                with col_e2:
                    st.metric("Farthest", f"{stats.get('max_distance_m', 0)} m")
                with col_e3:
                    st.metric("Avg Distance", f"{stats.get('avg_distance_m', 0)} m")
                
                min_dist = stats.get("min_distance_m", 0)
                max_dist = stats.get("max_distance_m", 0)
                if max_dist > 0 and max_dist < 500:
                    st.info(
                        f"**Camera Coverage:** {min_dist}m - {max_dist}m "
                        f"(~{max_dist - min_dist:.0f}m monitoring zone)"
                    )

        # show damage + reliability if available
        if st.session_state.session_log:
            latest = st.session_state.session_log[-1]

            st.markdown("---")
            st.subheader("🔧 Fatigue + Reliability (Simulated vs Observed)")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Simulation (LWR → Miner)**")
                st.metric("Damage (Miner)", f"{latest.get('sim_damage', 0.0):.6f}")
                st.metric("Reliability β", f"{latest.get('sim_beta')}" if latest.get("sim_beta") is not None else "—")
                st.metric("Shockwave", f"{latest.get('sim_shockwave', 0.0):.4f}")
            with c2:
                st.markdown("**Observed (YOLO load/speed → Miner)**")
                st.metric("Damage (Miner)", f"{latest.get('obs_damage', 0.0):.6f}")
                st.metric("Reliability β", f"{latest.get('obs_beta')}" if latest.get("obs_beta") is not None else "—")
                if vd.get("avg_speed_kph") is not None:
                    st.metric("Avg Speed", f"{vd.get('avg_speed_kph')} km/h")

        with st.expander("🧠 Confidence & Events", expanded=False):
            conf = compute_confidence_score(st.session_state.vehicle_data, st.session_state.weather, st.session_state.rf_metrics)
            st.metric("Confidence", f"{conf['label']} ({conf['score']})")
            st.write("Reasons:")
            for r in conf["reasons"]:
                st.write(f"- {r}")

            if st.session_state.session_log:
                ev = st.session_state.session_log[-1].get("events", [])
                if ev:
                    st.write("Events:")
                    st.write(", ".join(ev))
                else:
                    st.write("Events: None")

        if st.session_state.session_log:
            latest = st.session_state.session_log[-1]
            latest_beta = latest.get("beta_primary", None)
            st.metric("Reliability β (Primary)", f"{latest_beta:.2f}" if latest_beta is not None else "—")

            vd = latest.get("vehicle_data", {})

            st.markdown("---")
            st.subheader("🌊 Shockwave (Simulated vs Observed)")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**Simulated (LWR)**")
                st.metric("Shockwave speed", f"{float(latest.get('sim_shockwave', 0.0)):.4f}")

            with c2:
                st.markdown("**Observed (proxy)**")
                st.metric("Shockwave intensity", f"{float(latest.get('obs_shockwave', 0.0)):.3f}")

            with c3:
                st.markdown("**Observed inputs**")
                sp = vd.get("avg_speed_kph", None)
                st.metric("Avg Speed", f"{sp} km/h" if sp is not None else "N/A")
                st.metric("Density", f"{float(vd.get('density', 0.0)):.4f} veh/m")

            with st.expander("How to interpret this", expanded=False):
                st.write(
                    "- **Simulated shockwave** comes from the LWR density gradients and the chosen free-flow speed.\n"
                    "- **Observed shockwave** is a unitless proxy that increases when **speed drops** and/or **density rises quickly**.\n"
                    "- These two values are not the same units; they are shown together for **consistency checking and realism**."
                )

    # Auto-refresh live feed
    if auto_refresh_live and not st.session_state.monitoring_active:
        time.sleep(live_refresh_sec)
        st.rerun()
    
    if PLOTLY_AVAILABLE and st.session_state.session_log and len(st.session_state.session_log) >= 2:
        st.markdown("### 📈 Sim vs Observed Shockwave Speed")
        fig_sw = plot_shockwave_sim_vs_obs(st.session_state.session_log)
        st.plotly_chart(fig_sw, width='stretch')
    else:
        st.info("Need at least 2 captures to plot observed shockwave proxy.")

    st.markdown("---")
    st.subheader("📈 Sim vs Observed (Trends)")

    if PLOTLY_AVAILABLE and len(st.session_state.session_log) >= 2:
        fig = build_sim_vs_obs_fig(st.session_state.session_log)
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Need at least 2 captures to plot Sim vs Observed trends.")


    # =========================================================================
    # MONITORING SESSION
    # =========================================================================
    st.markdown("---")
    st.subheader("📹 Monitoring Session")

    col_ctrl1, col_ctrl2 = st.columns([2, 1])

    with col_ctrl1:
        # Auto-monitoring logic
        if st.session_state.auto_monitoring_enabled:
            if not st.session_state.monitoring_active and not st.session_state.auto_monitoring_started:
                st.session_state.monitoring_active = True
                st.session_state.monitoring_start_time = datetime.now()
                st.session_state.session_log = []
                st.session_state.auto_monitoring_started = True
                st.rerun()
        else:
            st.session_state.auto_monitoring_started = False

        if st.session_state.monitoring_active:
            start_time = st.session_state.monitoring_start_time
            if start_time is None:
                start_time = datetime.now()
                st.session_state.monitoring_start_time = start_time
            elapsed = (datetime.now() - start_time).total_seconds()
            exp_cfg = st.session_state.get("exp_settings", {}) or {}
            exp_active = bool(st.session_state.get("experiment_run_active", False))
            if st.session_state.validation_active:
                total_duration = st.session_state.validation_duration_min * 60
                interval_sec = st.session_state.validation_interval_sec
            elif exp_active:
                total_duration = float(exp_cfg.get("duration_min", monitor_duration)) * 60
                interval_sec = float(exp_cfg.get("interval_sec", capture_interval))
            else:
                total_duration = monitor_duration * 60
                interval_sec = capture_interval
            captures_done = len(st.session_state.session_log)
            expected_captures = int(total_duration / max(interval_sec, 0.01))
            
            progress = min(elapsed / total_duration, 1.0)
            st.progress(progress, text=f"Capture {captures_done}/{expected_captures} | {int(elapsed)}s / {total_duration}s")
            
            # Check if session complete
            if elapsed >= total_duration:
                st.session_state.monitoring_active = False
                st.session_state.validation_active = False
                st.success("✅ Monitoring session complete!")
                if st.session_state.get("experiment_run_active", False):
                    end_ts = datetime.now()
                    slog = st.session_state.get("session_log", []) or []
                    captures = len(slog)
                    avg_load = float(np.mean([float(e.get("vehicle_data", {}).get("load_tons", 0.0) or 0.0) for e in slog])) if captures else 0.0
                    betas = [e.get("beta_primary") for e in slog if e.get("beta_primary") is not None]
                    max_beta = float(max(betas)) if betas else None
                    beta_alerts = sum(1 for b in betas if float(b) < 3.0)
                    fatigue_vals = [e.get("combined_fatigue") for e in slog if e.get("combined_fatigue") is not None]
                    fatigue_alerts = sum(1 for f in fatigue_vals if float(f) >= 85.0)
                    alerts = int(beta_alerts + fatigue_alerts)

                    run_id = int(st.session_state.get("experiment_run_id", 0))
                    start_ts = st.session_state.get("monitoring_start_time") or end_ts
                    st.session_state.experiment_runs.append({
                        "run_id": run_id,
                        "start": start_ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "end": end_ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "captures": captures,
                        "avg_load_tons": round(avg_load, 2),
                        "max_beta": round(max_beta, 2) if max_beta is not None else None,
                        "alerts": alerts
                    })
                    st.session_state.experiment_run_active = False

    # Quick capture log (last 50)
    if st.session_state.session_log:
        st.markdown("#### 🗂️ Capture log")
        rows = []
        for e in st.session_state.session_log[-50:]:
            vd = e.get("vehicle_data", {}) or {}
            appr = vd.get("approaching", {}) or {}
            rows.append({
                "time": e.get("timestamp"),
                "vehicles": appr.get("total"),
                "load_tons": vd.get("load_tons"),
                "density_veh_per_m": vd.get("density"),
                "avg_speed_kph": vd.get("avg_speed_kph"),
                "beta_primary": e.get("beta_primary"),
                "obs_shockwave": e.get("obs_shockwave"),
                "sim_shockwave_speed": vd.get("sim_shockwave_speed"),
            })
        df_log = pd.DataFrame(rows)
        st.dataframe(df_log, width='stretch', hide_index=True)
        st.download_button(
            "⬇️ Download capture log (CSV)",
            data=df_log.to_csv(index=False).encode("utf-8"),
            file_name="capture_log.csv",
            mime="text/csv",
        )

    with st.expander("Speed Estimator Validation", expanded=False):
        if not st.session_state.session_log:
            st.info("Run a monitoring session to collect speed data.")
        else:
            log = st.session_state.session_log
            idx_max = max(0, len(log) - 1)
            if idx_max == 0:
                st.info("Only one frame available; using frame 0.")
                start_idx, end_idx = 0, 0
            else:
                start_idx, end_idx = st.slider(
                    "Frame range",
                    0,
                    idx_max,
                    (max(0, idx_max - 60), idx_max),
                )
            if end_idx <= start_idx:
                st.warning("Select a valid frame range.")
            else:
                window = log[start_idx:end_idx + 1]
                speeds = [e.get("vehicle_data", {}).get("avg_speed_mps") for e in window]
                speeds = [s for s in speeds if s is not None]
                if not speeds:
                    st.warning("No speed values available in this range.")
                else:
                    speeds_arr = np.asarray(speeds, dtype=float)
                    median_speed = float(np.median(speeds_arr))
                    spike_rate = float(np.mean(speeds_arr > 15.0)) * 100.0
                    stop_rate = float(np.mean(speeds_arr > 1.0)) * 100.0

                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        st.metric("Median speed (m/s)", f"{median_speed:.2f}")
                    with col_s2:
                        st.metric("Spike rate > 15 m/s", f"{spike_rate:.1f}%")
                    with col_s3:
                        st.metric("Stop false-motion > 1 m/s", f"{stop_rate:.1f}%")

                st.markdown("---")
                st.subheader("Threshold sweep (max_match_dist_px)")
                sweep_thresholds = [100, 150, 200, 250]
                sweep_rows = []
                usable_pairs = 0
                for i in range(start_idx + 1, end_idx + 1):
                    cur = log[i]
                    prev = log[i - 1]
                    if "detections" not in cur or "detections" not in prev:
                        continue
                    dt_s = cur.get("dt_s")
                    if dt_s is None:
                        t0 = prev.get("timestamp")
                        t1 = cur.get("timestamp")
                        if t0 and t1:
                            dt_s = (t1 - t0).total_seconds()
                    if not dt_s or dt_s <= 0:
                        continue
                    for thr in sweep_thresholds:
                        est = estimate_avg_speed_mps(
                            cur["detections"],
                            prev["detections"],
                            dt_s,
                            max_match_dist_px=float(thr),
                        )
                        if est is None:
                            continue
                        sweep_rows.append({"threshold_px": thr, "speed_mps": float(est)})
                    usable_pairs += 1

                if not sweep_rows:
                    st.info("Enable detection logging to run threshold sweep.")
                else:
                    df_sweep = pd.DataFrame(sweep_rows)
                    out = []
                    for thr in sweep_thresholds:
                        vals = df_sweep.loc[df_sweep["threshold_px"] == thr, "speed_mps"].to_numpy()
                        if vals.size == 0:
                            continue
                        out.append({
                            "threshold_px": thr,
                            "median_speed_mps": float(np.median(vals)),
                            "spike_rate_pct": float(np.mean(vals > 15.0)) * 100.0,
                            "n_pairs": int(vals.size),
                        })
                    st.dataframe(pd.DataFrame(out), width='stretch', hide_index=True)
                    st.caption(f"Usable pairs: {usable_pairs}")

                st.markdown("---")
                st.subheader("dt_s jitter sensitivity")
                if start_idx + 1 < end_idx:
                    dt_vals = []
                    for i in range(start_idx + 1, end_idx + 1):
                        cur = log[i]
                        prev = log[i - 1]
                        t0 = prev.get("timestamp")
                        t1 = cur.get("timestamp")
                        if t0 and t1:
                            dt_vals.append((t1 - t0).total_seconds())
                    if dt_vals:
                        dt_arr = np.asarray(dt_vals, dtype=float)
                        st.write(
                            f"dt_s median={np.median(dt_arr):.3f}s, "
                            f"p10={np.percentile(dt_arr,10):.3f}s, "
                            f"p90={np.percentile(dt_arr,90):.3f}s"
                        )
    
    # Monitoring loop
    if st.session_state.monitoring_active:
        start_time = st.session_state.monitoring_start_time
        if start_time is None:
            start_time = datetime.now()
            st.session_state.monitoring_start_time = start_time
        elapsed = (datetime.now() - start_time).total_seconds()
        if st.session_state.validation_active:
            total_duration = st.session_state.validation_duration_min * 60
            active_capture_interval = st.session_state.validation_interval_sec
        elif st.session_state.get("experiment_run_active", False):
            exp_cfg = st.session_state.get("exp_settings", {}) or {}
            total_duration = float(exp_cfg.get("duration_min", monitor_duration)) * 60
            active_capture_interval = float(exp_cfg.get("interval_sec", capture_interval))
        else:
            total_duration = monitor_duration * 60
            active_capture_interval = capture_interval
        
        if elapsed < total_duration:
            captures_done = len(st.session_state.session_log)
            next_capture_at = captures_done * active_capture_interval
            
            if elapsed >= next_capture_at:
                # Time to capture
                frame = capture_frame(camera_config["url"])
                
                if frame is not None:
                    now = datetime.now()
                    st.session_state.last_capture_time = now
                    exp_cfg = st.session_state.get("exp_settings", {}) or {}
                    exp_active = bool(st.session_state.get("experiment_run_active", False))
                    eff_conf = float(exp_cfg.get("confidence", confidence)) if exp_active else confidence
                    eff_use_roi = bool(exp_cfg.get("use_roi", st.session_state.get("use_roi", True)))
                    eff_roi_override = exp_cfg.get("roi", st.session_state.get("roi_override", None))
                    vehicle_data, annotated_frame, detections = detect_vehicles(
                                            frame=frame,
                                            model=st.session_state.yolo_model,  
                                            camera_config=camera_config,
                                            camera_name=selected_camera,
                                            lane_divider=lane_divider,
                                            confidence=eff_conf,
                                            bridge_config=bridge_config,
                                            use_roi=eff_use_roi,
                                            roi_override=eff_roi_override,
                                            weights=st.session_state.get("vehicle_weights", VEHICLE_WEIGHTS),
                                            focal_length_px=st.session_state.get("focal_length_px"),
                                        )
                    k = float(st.session_state.get("k_mpa_per_ton", 0.6))
                    stress_mpa = load_tons_to_stress_mpa(
                        vehicle_data.get("load_tons", 0.0),
                        k_mpa_per_ton=k
                    )
                    vehicle_data["stress_mpa_proxy"] = round(stress_mpa, 2)
                    # avg_speed_kph will be computed after speed estimation (if available)
                    prev_detections = st.session_state.get("prev_detections")
                    prev_time = st.session_state.get("prev_detections_time")
                    if prev_detections is not None and prev_time is not None:
                        dt_s = (now - prev_time).total_seconds()
                        if st.session_state.get("speed_mode") == "pixel":
                            avg_speed_mps = estimate_avg_speed_mps_pixel(
                                detections,
                                prev_detections,
                                dt_s,
                                meters_per_pixel=float(st.session_state.get("pixel_m_per_px", 0.05)),
                                axis=str(st.session_state.get("pixel_axis", "x")),
                                deadband_px=3.0,
                                dt_min_s=0.1,
                                dt_max_s=1.0,
                            )
                        else:
                            avg_speed_mps = estimate_avg_speed_mps(
                                detections,
                                prev_detections,
                                dt_s,
                                deadband_m=0.5,
                                dt_min_s=0.1,
                                dt_max_s=1.0,
                            )
                        if avg_speed_mps is not None:
                            ema_alpha = 0.3
                            prev_ema = st.session_state.get("speed_ema_mps")
                            if prev_ema is None:
                                st.session_state.speed_ema_mps = avg_speed_mps
                            else:
                                st.session_state.speed_ema_mps = (
                                    ema_alpha * avg_speed_mps + (1.0 - ema_alpha) * prev_ema
                                )
                            avg_speed_mps = st.session_state.speed_ema_mps
                        if avg_speed_mps is not None:
                            vehicle_data["avg_speed_mps"] = round(avg_speed_mps, 2)
                            vehicle_data["avg_speed_kph"] = round(avg_speed_mps * 3.6, 1)
                    st.session_state.prev_detections = detections
                    st.session_state.prev_detections_time = now
                    st.session_state.latest_detections = detections
                    
                    # -----------------------------
                    # REALISM SPLIT (Sim vs Observed)
                    # -----------------------------

                    # Observed signals from camera
                    density = float(vehicle_data.get("density", 0.03) or 0.03)
                    avg_speed_mps = vehicle_data.get("avg_speed_mps", None)

                    # Use observed speed to set LWR free-flow v_max (shockwave realism)
                    v_max_mps = choose_vmax_from_observed_speed(avg_speed_mps)

                    # Observed load (from YOLO counts × weights)
                    load_tons = float(vehicle_data.get("load_tons", 0.0) or 0.0)
                    stress_ref_mpa = float(st.session_state.get("stress_ref_mpa", 80.0))

                    # Simulated block: LWR -> Miner -> beta
                    sim_pack = compute_simulated_damage_beta(
                        density=density,
                        road_length_m=bridge_config.total_length_m,
                        v_max_mps=v_max_mps,
                        inject_jam=False,
                        miner_m=int(st.session_state.get("miner_m", 3)),
                        miner_C=float(st.session_state.get("miner_C", 1e12)),
                        mu_R=float(st.session_state.get("mu_R", 250.0)),
                        sigma_R=float(st.session_state.get("sigma_R", 25.0)),
                        stress_ref_mpa=stress_ref_mpa,
                    )

                    vehicle_data["sim_shockwave_speed"] = round(sim_pack["sim_shockwave"], 4)
                    vehicle_data["sim_fatigue"] = round(sim_pack["sim"]["fatigue"], 2)


                    obs_pack = compute_observed_damage_beta(
                        load_tons=load_tons,
                        density=density,
                        avg_speed_mps=avg_speed_mps,
                        miner_m=int(st.session_state.get("miner_m", 3)),
                        miner_C=float(st.session_state.get("miner_C", 1e12)),
                        mu_R=float(st.session_state.get("mu_R", 250.0)),
                        sigma_R=float(st.session_state.get("sigma_R", 25.0)),
                        stress_ref_mpa=stress_ref_mpa,
                    )

                    combined_fatigue = None
                    env_stress_value = None
                    if exp_active:
                        use_live_weather = bool(exp_cfg.get("use_live_weather", True))
                        weather = st.session_state.get("weather")
                        if use_live_weather or not weather:
                            weather = fetch_weather(bridge_config.latitude, bridge_config.longitude)
                            st.session_state.weather = weather
                        env_breakdown = calculate_environmental_stress(
                            weather.get("temperature", 10),
                            weather.get("humidity", 50),
                            weather.get("precipitation", 0),
                            weather.get("freeze_thaw_7day", 0),
                            weather.get("wind_speed", 5),
                            bridge_config.age_years
                        )
                        env_stress_value = env_breakdown.get("combined", None)
                        traffic_w = float(exp_cfg.get("traffic_weight", 0.70))
                        env_w = float(exp_cfg.get("environment_weight", 0.30))
                        combined_fatigue = traffic_w * float(sim_pack["sim"]["fatigue"]) + env_w * float(env_stress_value or 0.0)


                    # Log entry (store both)
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    st.session_state.latest_frame_rgb = frame_rgb
                    st.session_state.latest_frame_time = datetime.now()
                    pil_img = Image.fromarray(frame_rgb)
                    buffer = io.BytesIO()
                    pil_img.save(buffer, format="JPEG", quality=80)
                    img_b64 = base64.b64encode(buffer.getvalue()).decode()

                    entry = {
                        "timestamp": datetime.now(),
                        "vehicle_data": vehicle_data,
                        "image_b64": img_b64,
                        **sim_pack,
                        **obs_pack,
                    }
                    if st.session_state.get("log_detections_for_validation", False):
                        entry["detections"] = detections
                        entry["dt_s"] = dt_s
                    if combined_fatigue is not None:
                        entry["combined_fatigue"] = round(combined_fatigue, 1)
                    if env_stress_value is not None:
                        entry["env_stress"] = round(float(env_stress_value), 1)

                    # Compute observed shockwave proxy using previous capture
                    prev_entry = st.session_state.session_log[-2] if len(st.session_state.session_log) >= 2 else None
                    cur_density = float(entry.get("vehicle_data", {}).get("density_veh_per_m", 0.0) or 0.0)
                    cur_speed_mps = entry.get("vehicle_data", {}).get("avg_speed_mps", None)




                    # One headline reliability number for UI/report
                    entry["beta_primary"] = (
                        entry.get("obs_beta")
                        if entry.get("obs_beta") is not None
                        else entry.get("sim_beta")
                    )
                    # Observed shockwave proxy (explicit, explainable)
                    # NOTE: uses session_log[-1] vs session_log[-2], so append first OR compute with a temp list:
                    temp_log = st.session_state.session_log + [entry]
                    entry["obs_shockwave"] = compute_observed_shockwave_intensity(temp_log)
                    entry["obs_shockwave_speed"] = compute_observed_shockwave_speed_proxy(temp_log)


                    st.session_state.session_log.append(entry)

                    # Add events + gaps
                    latest_entry = st.session_state.session_log[-1]
                    latest_entry["events"] = detect_events(st.session_state.session_log)
                    latest_entry.update(compute_sim_obs_gap(latest_entry))

           
            # Auto-refresh for next capture (dynamic; respects the chosen interval).
            last_cap = st.session_state.get("last_capture_time", None)
            if last_cap is None:
                sleep_s = 0.5
            else:
                remaining = active_capture_interval - (datetime.now() - last_cap).total_seconds()
                sleep_s = max(0.1, min(0.5, remaining))
            time.sleep(sleep_s)
            st.rerun()
    
    with st.expander("Calibration (Advanced)", expanded=False):
        # Load->stress mapping
        k_mpa_per_ton = st.slider("k (MPa per ton)", 0.1, 2.0, 0.6, 0.1)
        stress_ref_mpa = st.slider("Stress reference (MPa)", 20.0, 200.0, 80.0, 5.0)

        # Miner damage params (proxy scale)
        miner_m = st.slider("Miner m", 2, 6, 3, 1)
        miner_C = st.number_input("Miner C", min_value=1e8, max_value=1e15, value=1e12, step=1e11, format="%.0e")

        # Reliability resistance model (proxy)
        mu_R = st.slider("μ_R (MPa)", 100.0, 600.0, 250.0, 10.0)
        sigma_R = st.slider("σ_R (MPa)", 5.0, 150.0, 25.0, 5.0)

        st.markdown("---")
        st.subheader("Focal Length Calibration (Distance Proxy)")
        st.caption("Estimate effective focal length using a known distance and observed bbox width.")

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            calib_class = st.selectbox("Vehicle class", ["car", "truck", "bus", "motorcycle"], index=0)
            known_distance_m = st.number_input("Known distance (m)", min_value=1.0, max_value=200.0, value=25.0, step=1.0)
        with col_f2:
            use_latest_bbox = st.checkbox("Use latest detection bbox width", value=True)
            bbox_width_px = st.number_input("BBox width (px)", min_value=1, max_value=2000, value=60, step=1)
        with col_f3:
            st.metric("Current f_px", f"{st.session_state.get('focal_length_px') or 'auto'}")

        if use_latest_bbox:
            latest = st.session_state.get("latest_detections") or []
            widths = [d.get("bbox_width_px") for d in latest if d.get("type") == calib_class and d.get("bbox_width_px")]
            if widths:
                bbox_width_px = int(np.median(widths))
            else:
                st.info("No matching detections found for selected class.")

        if st.button("Compute f_px"):
            if calib_class == "truck" or calib_class == "bus":
                real_width_m = CAMERA_CALIBRATION["typical_truck_width_m"]
            elif calib_class == "motorcycle":
                real_width_m = CAMERA_CALIBRATION["typical_motorcycle_width_m"]
            else:
                real_width_m = CAMERA_CALIBRATION["typical_car_width_m"]
            if bbox_width_px > 0 and known_distance_m > 0:
                f_px = (known_distance_m * float(bbox_width_px)) / float(real_width_m)
                st.session_state.focal_length_px = round(f_px, 2)
                st.success(f"Updated focal length: {st.session_state.focal_length_px} px")

    st.session_state.k_mpa_per_ton = k_mpa_per_ton
    st.session_state.stress_ref_mpa = stress_ref_mpa
    st.session_state.miner_m = miner_m
    st.session_state.miner_C = miner_C
    st.session_state.mu_R = mu_R
    st.session_state.sigma_R = sigma_R

    # Show session log
    if st.session_state.session_log:
        st.markdown("**Session Log:**")
        
        log_df = pd.DataFrame([
            {
                "Time": entry["timestamp"].strftime("%H:%M:%S"),
                "Incoming": entry["vehicle_data"]["approaching"]["total"],
                "Incoming Cars": entry["vehicle_data"]["approaching"]["car"],
                "Incoming Trucks": entry["vehicle_data"]["approaching"]["truck"],
                "Incoming Buses": entry["vehicle_data"]["approaching"]["bus"],
                "Outgoing": entry["vehicle_data"]["leaving"]["total"],
                "Outgoing Cars": entry["vehicle_data"]["leaving"]["car"],
                "Outgoing Trucks": entry["vehicle_data"]["leaving"]["truck"],
                "Outgoing Buses": entry["vehicle_data"]["leaving"]["bus"],
                "Load (tons)": entry["vehicle_data"]["load_tons"]
            }
            for entry in st.session_state.session_log
        ])
        st.dataframe(log_df, width='stretch', hide_index=True)

        # Activity log: vehicles passed (per capture)
        st.markdown("**Activity Log (Vehicles Passed)**")
        activity_df = pd.DataFrame([
            {
                "Time": entry["timestamp"].strftime("%H:%M:%S"),
                "Incoming Cars": entry["vehicle_data"]["approaching"]["car"],
                "Incoming Trucks": entry["vehicle_data"]["approaching"]["truck"],
                "Incoming Buses": entry["vehicle_data"]["approaching"]["bus"],
                "Incoming Total": entry["vehicle_data"]["approaching"]["total"],
                "Outgoing Cars": entry["vehicle_data"]["leaving"]["car"],
                "Outgoing Trucks": entry["vehicle_data"]["leaving"]["truck"],
                "Outgoing Buses": entry["vehicle_data"]["leaving"]["bus"],
                "Outgoing Total": entry["vehicle_data"]["leaving"]["total"],
            }
            for entry in st.session_state.session_log
        ])
        st.dataframe(activity_df, width='stretch', hide_index=True)

        # Export fatigue + beta log
        df_log = pd.DataFrame([
            {
                "timestamp": entry["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "vehicles": entry["vehicle_data"]["approaching"]["total"],
                "incoming": entry["vehicle_data"]["approaching"]["total"],
                "incoming_cars": entry["vehicle_data"]["approaching"]["car"],
                "incoming_trucks": entry["vehicle_data"]["approaching"]["truck"],
                "incoming_buses": entry["vehicle_data"]["approaching"]["bus"],
                "outgoing": entry["vehicle_data"]["leaving"]["total"],
                "outgoing_cars": entry["vehicle_data"]["leaving"]["car"],
                "outgoing_trucks": entry["vehicle_data"]["leaving"]["truck"],
                "outgoing_buses": entry["vehicle_data"]["leaving"]["bus"],
                "load_tons": entry["vehicle_data"]["load_tons"],
                "fatigue_damage_sim": entry.get("sim_damage", 0.0),
                "fatigue_damage_obs": entry.get("obs_damage", 0.0),
                "beta_sim": entry.get("sim_beta", None),
                "beta_obs": entry.get("obs_beta", None),
                "beta_primary": entry.get("beta_primary", None),
                "sim_damage": entry.get("sim_damage", 0.0),
                "sim_beta": entry.get("sim_beta", None),
                "sim_shockwave": entry.get("sim_shockwave", 0.0),
                "obs_damage": entry.get("obs_damage", 0.0),
                "obs_beta": entry.get("obs_beta", None),
                "stress_mpa_proxy": entry["vehicle_data"].get("stress_mpa_proxy", None),
                "gap_damage": entry.get("gap_damage", 0.0),
                "gap_beta": entry.get("gap_beta", None),
                "events": ",".join(entry.get("events", [])),
                "beta_primary": entry.get("beta_primary", None),
            }
            for entry in st.session_state.session_log


        ])
        if st.session_state.session_log:
            st.session_state.latest_beta = st.session_state.session_log[-1].get("beta_primary")
        csv_log = df_log.to_csv(index=False)

        st.download_button(
            "📁 Download Fatigue + β Log (CSV)",
            csv_log,
            f"fatigue_beta_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            width='stretch'
        )

        
        # Download session report
        if not st.session_state.monitoring_active:
            session_html = generate_session_report(
                bridge_config,
                st.session_state.session_log,
                st.session_state.weather
            )
            st.download_button(
                "📄 Download Session Report (HTML with Images)",
                session_html,
                f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                "text/html",
                width='stretch'
            )
        
            # Generate hourly summary
        df_log["hour"] = pd.to_datetime(df_log["timestamp"]).dt.strftime("%Y-%m-%d %H:00")

        hourly_summary = df_log.groupby("hour").agg({
            "vehicles": "sum",
            "load_tons": "mean",
            "damage_primary": "mean",
            "beta_primary": "mean"
        }).reset_index()

        df_log["damage_primary"] = np.where(
            df_log["fatigue_damage_obs"].notna(),
            df_log["fatigue_damage_obs"],
            df_log["fatigue_damage_sim"]
        )

        hourly_summary.rename(columns={
            "hour": "Hour",
            "vehicles": "Total Vehicles",
            "load_tons": "Avg Load (tons)",
            "fatigue_damage": "Avg Fatigue Damage",
            "reliability_index": "Avg Reliability Index (β)"
        }, inplace=True)

        csv_hourly = hourly_summary.to_csv(index=False)

        st.download_button(
            "📊 Download Hourly Summary (CSV)",
            csv_hourly,
            f"hourly_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            width='stretch'
        )

    
    # =========================================================================
    # ANALYZE BUTTON
    # =========================================================================
    st.markdown("---")
    
    if st.button("🔬 ANALYZE BASELINE CONDITIONS", type="primary", width='stretch'):
        
        with st.spinner("Running analysis with historical weather training..."):
            progress = st.progress(0)
            
            # Step 1: Fetch historical weather for training
            progress.progress(10, "Fetching historical weather data...")
            hist_df = fetch_historical_weather(bridge_config.latitude, bridge_config.longitude, months=12)
            
            if hist_df is None or len(hist_df) == 0:
                st.error("Failed to fetch historical weather. Using fallback.")
                hist_df = None
            else:
                st.session_state.historical_weather = hist_df
                st.session_state.historical_analysis = analyze_historical_weather(hist_df)
            
            # Step 2: Generate training data from historical weather + traffic scenarios
            progress.progress(30, "Generating training data from historical patterns...")
            
            if hist_df is not None:
                training_data = generate_training_data_from_historical(
                    hist_df, 
                    road_length_m=bridge_config.total_length_m,
                    scenarios_per_day=5  # 365 days × 5 = ~1825 training samples
                )
                st.session_state.mc_data = training_data
            else:
                # Fallback to simple Monte Carlo
                density = st.session_state.vehicle_data.get("density", 0.03)
                training_data = run_monte_carlo(mc_runs, bridge_config.total_length_m, density, jam_probability)
                st.session_state.mc_data = training_data
            
            # Step 3: Train model on historical + traffic data
            progress.progress(50, "Training ML model on historical patterns...")
            model, metrics = train_model_cv(training_data)
            st.session_state.rf_model = model
            st.session_state.rf_metrics = metrics
            
            # Step 4: Predict using CURRENT weather + CURRENT traffic
            progress.progress(70, "Predicting with current conditions...")
            
            density = st.session_state.vehicle_data.get("density", 0.03)
            weather = st.session_state.weather
            
            # Calculate shockwave from current density
            current_sim = run_lwr_simulation(
                initial_density=density,
                road_length_m=bridge_config.total_length_m,
                v_max_mps=22.2
            )
            
            # Get current truck percentage from detection
            approaching = st.session_state.vehicle_data.get("approaching", {})
            total = approaching.get("total", 1)
            trucks = approaching.get("truck", 0)
            truck_pct = trucks / total if total > 0 else 0.15
            
            # REAL-TIME PREDICTION using both traffic AND weather
            traffic_fatigue, used_features = predict_fatigue_with_weather(
                model=model,
                density=density,
                shockwave=current_sim["shockwave_speed"],
                truck_pct=truck_pct,
                temperature=weather.get("temperature", 15),
                precipitation=weather.get("precipitation", 0),
                wind_speed=weather.get("wind_speed", 10),
                freeze_thaw=weather.get("freeze_thaw_7day", 0),
                month=datetime.now().month
            )
            
            st.session_state.baseline_traffic_fatigue = traffic_fatigue
            
            # Environmental stress (for display breakdown)
            env_stress = calculate_environmental_stress(
                weather["temperature"],
                weather["humidity"],
                weather["precipitation"],
                weather["freeze_thaw_7day"],
                weather.get("wind_speed", 10),
                bridge_config.age_years
            )
            st.session_state.baseline_env_stress = env_stress["combined"]
            st.session_state.env_breakdown = env_stress
            
            progress.progress(100, "Done!")
            st.session_state.baseline_calculated = True
            time.sleep(0.3)
            progress.empty()
        
        # DEBUG OUTPUT IN UI
        st.success("✅ Baseline analysis complete!")
        
        # Show training info
        if st.session_state.rf_metrics:
            metrics = st.session_state.rf_metrics
            with st.expander("🔍 DEBUG: Model Training Details", expanded=True):
                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    st.metric("Training Samples", metrics.get("samples", "N/A"))
                with col_d2:
                    st.metric("CV R² (mean)", f"{metrics.get('cv_r2_mean', 0):.3f}")
                with col_d3:
                    st.metric("CV MAE (mean)", f"{metrics.get('cv_mae_mean', 0):.2f}")
                    
                
                st.markdown("**Features Used:**")
                st.write(metrics.get("features_used", []))
                
                st.markdown("**Feature Importance (Top 5):**")
                importance = metrics.get("feature_importance_mean", {})
                for i, (feat, imp) in enumerate(list(importance.items())[:5]):
                    st.write(f"{i+1}. **{feat}**: {imp:.3f}")
                
                st.markdown("**Current Conditions Used for Prediction:**")
                st.json({
                    "density": round(density, 4),
                    "truck_pct": round(truck_pct, 2),
                    "temperature": weather.get("temperature"),
                    "precipitation": weather.get("precipitation"),
                    "wind_speed": weather.get("wind_speed"),
                    "freeze_thaw": weather.get("freeze_thaw_7day"),
                    "month": datetime.now().month
                })
    
    # =========================================================================
    # SCENARIO DASHBOARD
    # =========================================================================
    if st.session_state.baseline_calculated:
        st.markdown("---")
        st.header("🎛️ Scenario Dashboard")
        st.markdown("*Adjust parameters to see how fatigue score changes*")
        
        col_sliders, col_result = st.columns([1, 1])
        
        with col_sliders:
            st.subheader("Traffic Scenarios")
            traffic_change = st.slider(
                "Traffic Volume Change",
                min_value=-30,
                max_value=50,
                value=0,
                step=5,
                format="%+d%%"
            )
            traffic_mult = 1 + traffic_change / 100
            
            truck_pct = st.slider(
                "Truck Percentage",
                min_value=5,
                max_value=40,
                value=15,
                step=5,
                format="%d%%"
            ) / 100
            
            st.subheader("Environmental Scenarios")
            scenario_temp = st.slider(
                "Temperature (°C)",
                min_value=-20,
                max_value=40,
                value=int(st.session_state.weather.get("temperature", 10))
            )
            
            scenario_freeze = st.slider(
                "Freeze-Thaw Cycles (weekly)",
                min_value=0,
                max_value=14,
                value=st.session_state.weather.get("freeze_thaw_7day", 0)
            )
            
            scenario_precip = st.slider(
                "Precipitation (mm)",
                min_value=0,
                max_value=50,
                value=int(st.session_state.weather.get("precipitation", 0))
            )
        
        # Calculate scenario
        scenario = calculate_scenario_fatigue(
            st.session_state.baseline_traffic_fatigue,
            st.session_state.baseline_env_stress,
            traffic_mult,
            truck_pct,
            scenario_freeze,
            scenario_temp,
            scenario_precip,
            bridge_config.age_years
        )
        bridge_age = bridge_config.age_years
        with col_result:
            st.subheader("Fatigue Prediction")
            
            status_text, _ = get_status(scenario["combined_fatigue"])
            
            # Big score display
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                            padding: 30px; border-radius: 15px; text-align: center; color: white;">
                    <div style="font-size: 4em; font-weight: bold;">{scenario['combined_fatigue']}</div>
                    <div style="font-size: 1.2em;">/ 100</div>
                    <div style="font-size: 1.5em; margin-top: 10px;">{status_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("")
            
            col_t, col_e = st.columns(2)
            with col_t:
                st.metric("Traffic Stress", f"{scenario['traffic_fatigue']:.1f}", 
                         delta=f"{scenario['traffic_fatigue'] - st.session_state.baseline_traffic_fatigue:.1f}")
            with col_e:
                st.metric("Environmental", f"{scenario['environmental_stress']:.1f}",
                         delta=f"{scenario['environmental_stress'] - st.session_state.baseline_env_stress:.1f}")

        # Handle experiment runs (single-shot)
        if st.session_state.get("run_single_experiment"):
            exp_density = float(st.session_state.get("exp_density", 0.05))
            exp_vmax_kph = float(st.session_state.get("exp_vmax", 80))
            exp_jam = bool(st.session_state.get("exp_jam", False))

            sim = run_lwr_simulation(
                initial_density=exp_density,
                road_length_m=bridge_config.total_length_m,
                v_max_mps=exp_vmax_kph / 3.6,
                inject_jam=exp_jam
            )
            sim_pack = compute_simulated_damage_beta(
                density=exp_density,
                road_length_m=bridge_config.total_length_m,
                v_max_mps=exp_vmax_kph / 3.6,
                inject_jam=exp_jam,
                miner_m=int(st.session_state.get("miner_m", 3)),
                miner_C=float(st.session_state.get("miner_C", 1e12)),
                mu_R=float(st.session_state.get("mu_R", 250.0)),
                sigma_R=float(st.session_state.get("sigma_R", 25.0)),
                stress_ref_mpa=float(st.session_state.get("stress_ref_mpa", 80.0)),
            )

            st.session_state.last_experiment = {"sim": sim, "sim_pack": sim_pack}
            st.session_state.experiment_log.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "density": exp_density,
                "v_max": exp_vmax_kph,
                "jam": exp_jam,
                "fatigue": sim.get("fatigue"),
                "beta": sim_pack.get("sim_beta")
            })
            st.session_state.run_single_experiment = False
        
        # Charts
        st.markdown("---")
        st.subheader("📈 Analysis Charts")
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Fatigue Breakdown", 
            "Sensitivity", 
            "Environmental Factors",
            "📅 Historical Weather",
            "📊 NYSDOT Validation",
            "📉 Reliability Index",
            "🧪 Experiments"
        ])
        
        with tab1:
            fig = create_fatigue_breakdown_chart(scenario["traffic_fatigue"], scenario["environmental_stress"])
            st.plotly_chart(fig, width='stretch')
        
        with tab2:
            fig = create_sensitivity_chart(st.session_state.baseline_traffic_fatigue, scenario["environmental_stress"])
            st.plotly_chart(fig, width='stretch')
        
        with tab3:
            fig = create_environmental_breakdown_chart(scenario["env_breakdown"])
            st.plotly_chart(fig, width='stretch')
        
        with tab4:
            st.markdown("### 📅 Historical Weather Analysis (Last 12 Months)")
            st.markdown("*Analyze seasonal patterns that affect bridge fatigue*")
            
            if st.button("🔄 Load Historical Weather Data", key="load_historical"):
                with st.spinner("Fetching 12 months of weather data..."):
                    hist_df = fetch_historical_weather(bridge_config.latitude, bridge_config.longitude, months=12)
                    if hist_df is not None:
                        st.session_state.historical_weather = hist_df
                        st.session_state.historical_analysis = analyze_historical_weather(hist_df)
                        st.success("✅ Historical data loaded!")
                    else:
                        st.error("Failed to fetch historical data")
            
            if "historical_analysis" in st.session_state and st.session_state.historical_analysis:
                analysis = st.session_state.historical_analysis
                
                # Summary metrics
                col_h1, col_h2, col_h3, col_h4 = st.columns(4)
                with col_h1:
                    st.metric("Total Freeze-Thaw Cycles", analysis["total_freeze_thaw"])
                with col_h2:
                    st.metric("Total Precipitation", f"{analysis['total_precipitation_mm']:.0f} mm")
                with col_h3:
                    st.metric("Salt Exposure", f"{analysis['total_salt_exposure_mm']:.0f} mm")
                with col_h4:
                    st.metric("Winter Contribution", f"{analysis['winter_corrosion_contribution']:.0f}%")
                
                # Charts
                if "monthly_data" in analysis and len(analysis["monthly_data"]) > 0:
                    fig1, fig2 = create_historical_weather_charts(analysis["monthly_data"])
                    st.plotly_chart(fig1, width='stretch')
                    st.plotly_chart(fig2, width='stretch')
                
                if "seasonal_data" in analysis:
                    fig3 = create_seasonal_fatigue_chart(analysis["seasonal_data"])
                    st.plotly_chart(fig3, width='stretch')
                
                # Key finding for paper
                st.info(
                    f"**📝 Key Finding for Paper:**\n\n"
                    f"\"Analysis of 12-month historical weather data shows {analysis['total_freeze_thaw']} "
                    f"freeze-thaw cycles annually, with winter months contributing "
                    f"{analysis['winter_corrosion_contribution']:.0f}% of total corrosion exposure due to "
                    f"combined precipitation and road salt application.\""
                )
        
        with tab5:
            st.markdown("### 📊 NYSDOT Traffic Count Validation")
            st.markdown("*Compare your detection results with official NYSDOT data*")
            
            if st.session_state.session_log:
                st.success(f"✅ Session data available: {len(st.session_state.session_log)} captures")
                
                # Generate comparison report
                comparison_html = generate_nysdot_comparison_report(
                    bridge_config,
                    st.session_state.session_log
                )
                
                st.download_button(
                    "📥 Download NYSDOT Comparison Template (HTML)",
                    comparison_html,
                    f"nysdot_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    "text/html",
                    width='stretch'
                )
                
                st.markdown("---")
                st.markdown("**Instructions:**")
                st.markdown("""
                1. Download the comparison template above
                2. Open [NYSDOT Traffic Data Viewer](https://gisportalny.dot.ny.gov/portalny/apps/webappviewer/index.html?id=28537cbc8b5941e19cf8e959b16797b4)
                3. Find I-87 near Twin Bridges (Cohoes, NY) Station 153000
                4. Click on Continuous Count station
                5. Download hourly data for your session date
                6. Enter NYSDOT counts in the template
                7. Click "Calculate Accuracy" to see comparison
                """)
                
                # Quick session stats
                total_detected = sum(e["vehicle_data"]["approaching"]["total"] for e in st.session_state.session_log)
                duration_hrs = (st.session_state.session_log[-1]["timestamp"] - st.session_state.session_log[0]["timestamp"]).total_seconds() / 3600
                
                st.markdown("**Your Session Summary:**")
                col_n1, col_n2, col_n3 = st.columns(3)
                with col_n1:
                    st.metric("Total Detected", total_detected)
                with col_n2:
                    st.metric("Duration", f"{duration_hrs:.1f} hrs")
                with col_n3:
                    st.metric("Rate", f"{total_detected/max(duration_hrs, 0.1):.0f} veh/hr")
            else:
                st.warning("⚠️ No session data yet. Start a monitoring session first to generate comparison data.")

        with tab6:
            st.subheader("📉 Reliability Index Over Time")
            fig_beta = plot_reliability_over_time(st.session_state.session_log)
            st.plotly_chart(fig_beta, width='stretch')
            if st.session_state.session_log and len(st.session_state.session_log) >= 2:
                st.plotly_chart(plot_sim_obs_over_time(st.session_state.session_log), width='stretch')

        with tab7:
            st.subheader("📊 Monte Carlo Results Summary")

            mc_df = st.session_state.get("mc_data")
            if mc_df is not None and len(mc_df) > 0:
                summary = mc_df.describe()[["density", "shockwave_speed", "fatigue"]]
                st.dataframe(summary, width='stretch')

                percentiles = mc_df[["fatigue"]].quantile([0.1, 0.5, 0.9]).reset_index()
                percentiles.columns = ["Percentile", "Fatigue"]
                st.dataframe(percentiles, hide_index=True)

                if PLOTLY_AVAILABLE:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=mc_df["density"],
                        y=mc_df["fatigue"],
                        mode="markers",
                        opacity=0.5,
                        name="MC runs"
                    ))
                    fig.update_layout(
                        title="Fatigue vs Traffic Density",
                        xaxis_title="Density (veh/m)",
                        yaxis_title="Fatigue Score",
                        template="plotly_white",
                        height=350
                    )
                    st.plotly_chart(fig, width='stretch')

                    fig = go.Figure()
                    fig.add_histogram(
                        x=mc_df["fatigue"],
                        nbinsx=30,
                        name="Fatigue distribution"
                    )
                    fig.add_vline(x=50, line_dash="dash", annotation_text="Safe")
                    fig.add_vline(x=70, line_dash="dash", annotation_text="Monitor")
                    fig.add_vline(x=85, line_dash="dash", annotation_text="Critical")

                    fig.update_layout(
                        title="Fatigue Distribution Across Monte Carlo Runs",
                        xaxis_title="Fatigue Score",
                        yaxis_title="Frequency",
                        template="plotly_white",
                        height=350
                    )
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.warning("Plotly not available; install plotly to render charts.")
            else:
                st.info("Run baseline analysis to generate Monte Carlo data.")

            st.markdown("---")
            comparison = pd.DataFrame([
                {
                    "Scenario": "Baseline",
                    "Traffic Fatigue": st.session_state.baseline_traffic_fatigue,
                    "Environmental Stress": st.session_state.baseline_env_stress,
                    "Combined": (
                        0.7 * st.session_state.baseline_traffic_fatigue +
                        0.3 * st.session_state.baseline_env_stress
                    )
                },
                {
                    "Scenario": "What-if",
                    "Traffic Fatigue": scenario["traffic_fatigue"],
                    "Environmental Stress": scenario["environmental_stress"],
                    "Combined": scenario["combined_fatigue"]
                }
            ])
            st.dataframe(comparison, width='stretch')

            st.subheader("🧾 Experiment Log")
            if st.session_state.experiment_log:
                st.dataframe(pd.DataFrame(st.session_state.experiment_log), width='stretch')
            else:
                st.info("No experiments yet. Enable Experiment Mode in the sidebar to run one.")

        

        # =====================================================================
        # DOWNLOADS
        # =====================================================================
        st.markdown("---")
        st.subheader("📥 Download Reports")
        
        col_d1, col_d2, col_d3 = st.columns(3)
        
        with col_d1:
            html_report = generate_html_report(
                bridge_config,
                st.session_state.vehicle_data,
                st.session_state.weather,
                scenario,
                st.session_state.rf_metrics
            )
            st.download_button(
                "📄 Download HTML Report",
                html_report,
                f"bridge_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                "text/html",
                width='stretch'
            )
        
        with col_d2:
            if st.session_state.mc_data is not None:
                csv = st.session_state.mc_data.to_csv(index=False)
                st.download_button(
                    "📊 Download MC Data (CSV)",
                    csv,
                    f"monte_carlo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    width='stretch'
                )
        
        with col_d3:
            # Calculate avg_beta for summary
            if st.session_state.session_log:
                avg_beta = sum(e.get("reliability_index", 0.0) for e in st.session_state.session_log) / len(st.session_state.session_log)
            else:
                avg_beta = 0.0

            # Summary text
            summary = f"""
HYBRID DIGITAL TWIN - BRIDGE FATIGUE ASSESSMENT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Bridge: {bridge_config.name}
Age: {bridge_config.age_years} years

CURRENT CONDITIONS:
- Vehicles on bridge: {st.session_state.vehicle_data.get('approaching', {}).get('total', 'N/A')}
- Load: {st.session_state.vehicle_data.get('load_tons', 'N/A')} tons
- Temperature: {st.session_state.weather.get('temperature')}°C
- Freeze-thaw cycles (7d): {st.session_state.weather.get('freeze_thaw_7day')}

FATIGUE ASSESSMENT:
- Traffic stress: {scenario['traffic_fatigue']}/100
- Environmental stress: {scenario['environmental_stress']}/100
- Combined fatigue: {scenario['combined_fatigue']}/100
- Status: {status_text}
- Avg Reliability Index β: {avg_beta:.2f}


LIMITATIONS:
This is a proof-of-concept. Fatigue scores are proxy metrics,
not validated against actual structural monitoring data.
            """
            st.download_button(
                "📝 Download Summary (TXT)",
                summary,
                f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain",
                width='stretch'
            )


if __name__ == "__main__":
    main()
