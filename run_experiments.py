from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import core functions from your Streamlit app module.
# Safe because main.py only runs Streamlit UI under: if __name__ == "__main__": main()
from main import (  # type: ignore
    run_lwr_simulation,
    run_monte_carlo,
    capture_frame,
    detect_vehicles,
    estimate_avg_speed_mps,
    VEHICLE_WEIGHTS,
    BridgeConfig,
    CAMERAS,
    calculate_environmental_stress,
    compute_reliability_index,
    load_tons_to_stress_mpa,
)


# -----------------------------
# Experiment definitions
# -----------------------------

@dataclass
class SweepSpec:
    name: str
    road_length_m: float = 1768.0
    rho_max: float = 0.20
    inject_jam_probability: float = 0.30
    n_reps: int = 50               # repetitions per point for robustness
    seed: int = 42

    # domain for sweeps (overridden per experiment)
    densities: Tuple[float, ...] = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12)  # vehicles/m
    v_max_kmh: Tuple[float, ...] = (40, 60, 80, 100)                    # km/h


def _ensure_outdir(outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figs").mkdir(parents=True, exist_ok=True)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    return outdir


def _set_seed(seed: int) -> None:
    np.random.seed(seed)


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except Exception:
            return None
    return None


def _single_point_stats(
    density: float,
    v_max_kmh: float,
    road_length_m: float,
    rho_max: float,
    inject_jam_probability: float,
    n_reps: int,
) -> Dict[str, float]:
    """Run n_reps sims at a single (density, v_max) point and return summary stats."""
    shock = []
    fat = []
    jam_flags = []
    for _ in range(n_reps):
        inject_jam = bool(np.random.random() < inject_jam_probability)
        sim = run_lwr_simulation(
            initial_density=float(density),
            road_length_m=float(road_length_m),
            v_max_mps=float(v_max_kmh) / 3.6,
            inject_jam=inject_jam,
            rho_max=float(rho_max),
        )
        shock.append(float(sim.get("shockwave_speed", 0.0)))
        fat.append(float(sim.get("fatigue", 0.0)))
        jam_flags.append(1 if inject_jam else 0)

    shock = np.asarray(shock, dtype=float)
    fat = np.asarray(fat, dtype=float)
    jam_rate = float(np.mean(jam_flags)) if len(jam_flags) else 0.0

    return {
        "shockwave_mean": float(np.mean(shock)),
        "shockwave_p50": float(np.percentile(shock, 50)),
        "shockwave_p90": float(np.percentile(shock, 90)),
        "fatigue_mean": float(np.mean(fat)),
        "fatigue_p50": float(np.percentile(fat, 50)),
        "fatigue_p90": float(np.percentile(fat, 90)),
        "jam_rate": jam_rate,
    }


def sweep_density(spec: SweepSpec) -> pd.DataFrame:
    """Sweep initial density at fixed v_max=80 km/h."""
    rows: List[Dict] = []
    v_max = 80.0
    for rho in spec.densities:
        stats = _single_point_stats(
            density=rho,
            v_max_kmh=v_max,
            road_length_m=spec.road_length_m,
            rho_max=spec.rho_max,
            inject_jam_probability=spec.inject_jam_probability,
            n_reps=spec.n_reps,
        )
        rows.append(
            {
                "experiment": spec.name,
                "sweep": "density",
                "density": rho,
                "v_max_kmh": v_max,
                "road_length_m": spec.road_length_m,
                "rho_max": spec.rho_max,
                "inject_jam_probability": spec.inject_jam_probability,
                **stats,
            }
        )
    return pd.DataFrame(rows)


def sweep_vmax(spec: SweepSpec) -> pd.DataFrame:
    """Sweep v_max at fixed density=0.06 vehicles/m."""
    rows: List[Dict] = []
    rho = 0.06
    for v_max in spec.v_max_kmh:
        stats = _single_point_stats(
            density=rho,
            v_max_kmh=float(v_max),
            road_length_m=spec.road_length_m,
            rho_max=spec.rho_max,
            inject_jam_probability=spec.inject_jam_probability,
            n_reps=spec.n_reps,
        )
        rows.append(
            {
                "experiment": spec.name,
                "sweep": "v_max",
                "density": rho,
                "v_max_kmh": float(v_max),
                "road_length_m": spec.road_length_m,
                "rho_max": spec.rho_max,
                "inject_jam_probability": spec.inject_jam_probability,
                **stats,
            }
        )
    return pd.DataFrame(rows)


def sweep_jam_probability(spec: SweepSpec, jam_probs=(0.0, 0.1, 0.3, 0.5, 0.7)) -> pd.DataFrame:
    """Sweep jam injection probability at fixed density=0.06, v_max=80."""
    rows: List[Dict] = []
    rho = 0.06
    v_max = 80.0
    for p in jam_probs:
        stats = _single_point_stats(
            density=rho,
            v_max_kmh=v_max,
            road_length_m=spec.road_length_m,
            rho_max=spec.rho_max,
            inject_jam_probability=float(p),
            n_reps=spec.n_reps,
        )
        rows.append(
            {
                "experiment": spec.name,
                "sweep": "jam_probability",
                "density": rho,
                "v_max_kmh": v_max,
                "road_length_m": spec.road_length_m,
                "rho_max": spec.rho_max,
                "inject_jam_probability": float(p),
                **stats,
            }
        )
    return pd.DataFrame(rows)


def grid_heatmap(spec: SweepSpec) -> pd.DataFrame:
    """
    2D grid sweep (density x v_max) that is perfect for heatmaps.
    """
    rows: List[Dict] = []
    for rho in spec.densities:
        for v_max in spec.v_max_kmh:
            stats = _single_point_stats(
                density=float(rho),
                v_max_kmh=float(v_max),
                road_length_m=spec.road_length_m,
                rho_max=spec.rho_max,
                inject_jam_probability=spec.inject_jam_probability,
                n_reps=spec.n_reps,
            )
            rows.append(
                {
                    "experiment": spec.name,
                    "sweep": "grid",
                    "density": float(rho),
                    "v_max_kmh": float(v_max),
                    "road_length_m": spec.road_length_m,
                    "rho_max": spec.rho_max,
                    "inject_jam_probability": spec.inject_jam_probability,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def monte_carlo_bundle(spec: SweepSpec, num_runs: int = 500, live_density: Optional[float] = None) -> pd.DataFrame:
    """
    Runs your existing Monte Carlo function and returns the raw samples.
    Perfect for histograms / scatter plots.
    """
    df = run_monte_carlo(
        num_runs=int(num_runs),
        road_length_m=float(spec.road_length_m),
        live_density=None if live_density is None else float(live_density),
        inject_jam_probability=float(spec.inject_jam_probability),
        rho_max=float(spec.rho_max),
        seed=int(spec.seed),
    )
    df.insert(0, "experiment", spec.name)
    df.insert(1, "sweep", "monte_carlo")
    if "road_length_m" not in df.columns:
        df["road_length_m"] = float(spec.road_length_m)
    if "rho_max" not in df.columns:
        df["rho_max"] = float(spec.rho_max)
    df["inject_jam_probability"] = float(spec.inject_jam_probability)
    return df


# -----------------------------
# Live data sources (headless)
# -----------------------------

def _read_latest_csv_row(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    return df.iloc[-1].to_dict()


def _write_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def run_live_csv(
    input_csv: Path,
    output_csv: Path,
    duration_sec: float,
    interval_sec: float,
    density_col: str,
    speed_col: Optional[str] = None,
    load_col: Optional[str] = None,
    env_temp_col: Optional[str] = None,
    env_humidity_col: Optional[str] = None,
    env_precip_col: Optional[str] = None,
    env_freeze_thaw_col: Optional[str] = None,
    env_wind_col: Optional[str] = None,
    v_max_kmh_default: float = 80.0,
    road_length_m: float = 1768.0,
    rho_max: float = 0.20,
    k_mpa_per_ton: float = 0.6,
    env_temperature: float = 10.0,
    env_humidity: float = 50.0,
    env_precipitation: float = 0.0,
    env_freeze_thaw_7day: float = 0.0,
    env_wind_speed: float = 5.0,
    bridge_age_years: float = 99.0,
    mu_R: float = 250.0,
    sigma_R: float = 25.0,
    sigma_S_ratio: float = 0.10,
    randomize_constants: bool = True,
) -> None:
    end_ts = time.time() + duration_sec
    last_seen = None
    while time.time() < end_ts:
        row = _read_latest_csv_row(input_csv)
        if row is None:
            time.sleep(interval_sec)
            continue
        if last_seen is not None and row == last_seen:
            time.sleep(interval_sec)
            continue
        last_seen = dict(row)

        density = _safe_float(row.get(density_col)) or 0.0
        speed_kph = None
        if speed_col:
            speed_kph = _safe_float(row.get(speed_col))
        v_max_mps = float(((speed_kph if speed_kph is not None else v_max_kmh_default)) / 3.6)

        rho_max_run = float(np.random.uniform(0.18, 0.22)) if randomize_constants else float(rho_max)
        road_length_run = float(road_length_m) * (float(np.random.uniform(0.98, 1.02)) if randomize_constants else 1.0)
        k_mpa_per_ton_run = float(np.random.uniform(0.25, 0.35)) if randomize_constants else float(k_mpa_per_ton)

        load_tons = _safe_float(row.get(load_col)) if load_col else None
        stress_mpa_proxy = None
        if load_tons is not None:
            stress_mpa_proxy = float(load_tons_to_stress_mpa(load_tons, k_mpa_per_ton_run))

        temperature = _safe_float(row.get(env_temp_col)) if env_temp_col else env_temperature
        humidity = _safe_float(row.get(env_humidity_col)) if env_humidity_col else env_humidity
        precipitation = _safe_float(row.get(env_precip_col)) if env_precip_col else env_precipitation
        freeze_thaw = _safe_float(row.get(env_freeze_thaw_col)) if env_freeze_thaw_col else env_freeze_thaw_7day
        wind_speed = _safe_float(row.get(env_wind_col)) if env_wind_col else env_wind_speed

        env_breakdown = calculate_environmental_stress(
            temperature=temperature or 0.0,
            humidity=humidity or 0.0,
            precipitation=precipitation or 0.0,
            freeze_thaw_cycles=int(freeze_thaw or 0.0),
            wind_speed=wind_speed or 0.0,
            bridge_age=int(bridge_age_years),
        )
        env_stress = float(env_breakdown.get("combined", 0.0))
        if randomize_constants:
            env_stress = float(np.clip(np.random.normal(env_stress, 2.0), 0.0, 100.0))

        sim = run_lwr_simulation(
            initial_density=density,
            road_length_m=road_length_run,
            v_max_mps=v_max_mps,
            inject_jam=False,
            rho_max=rho_max_run,
        )

        reliability_beta = None
        if stress_mpa_proxy is not None:
            sigma_S = max(1e-6, abs(stress_mpa_proxy) * sigma_S_ratio)
            reliability_beta = float(
                compute_reliability_index(
                    mu_R=float(mu_R),
                    sigma_R=float(sigma_R),
                    mu_S=float(stress_mpa_proxy),
                    sigma_S=float(sigma_S),
                )
            )

        out_row = {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "density": density,
            "speed_kph": speed_kph,
            "v_max_kph_used": v_max_mps * 3.6,
            "load_tons": load_tons,
            "stress_mpa_proxy": stress_mpa_proxy,
            "reliability_beta": reliability_beta,
            "env_stress": env_stress,
            "sim_fatigue": sim.get("fatigue"),
            "sim_shockwave_speed": sim.get("shockwave_speed"),
            "avg_density": sim.get("avg_density"),
            "max_stress": sim.get("max_stress"),
            "k_mpa_per_ton": k_mpa_per_ton_run,
            "rho_max": rho_max_run,
            "road_length_m": road_length_run,
            "source": "live_csv",
        }
        _write_row(output_csv, out_row)
        time.sleep(interval_sec)


def run_live_camera(
    output_csv: Path,
    duration_sec: float,
    interval_sec: float,
    camera_name: str,
    confidence: float = 0.15,
    lane_divider: float = 0.43,
    use_roi: bool = True,
    v_max_kmh_default: float = 80.0,
    rho_max: float = 0.20,
    k_mpa_per_ton: float = 0.6,
    env_temperature: float = 10.0,
    env_humidity: float = 50.0,
    env_precipitation: float = 0.0,
    env_freeze_thaw_7day: float = 0.0,
    env_wind_speed: float = 5.0,
    mu_R: float = 250.0,
    sigma_R: float = 25.0,
    sigma_S_ratio: float = 0.10,
    randomize_constants: bool = True,
) -> None:
    bridge_config = BridgeConfig()
    camera_cfg = CAMERAS[camera_name]
    model = None
    try:
        from ultralytics import YOLO  # type: ignore
        model = YOLO("yolov8n.pt")
    except Exception:
        raise RuntimeError("YOLO not available. Install ultralytics and ensure yolov8n.pt is present.")

    prev_detections = None
    prev_time = None
    end_ts = time.time() + duration_sec

    while time.time() < end_ts:
        frame = capture_frame(camera_cfg["url"])
        if frame is None:
            time.sleep(interval_sec)
            continue

        vehicle_data, _, detections = detect_vehicles(
            frame=frame,
            model=model,
            camera_config=camera_cfg,
            camera_name=camera_name,
            lane_divider=lane_divider,
            confidence=confidence,
            bridge_config=bridge_config,
            use_roi=use_roi,
            roi_override=None,
            weights={k: float(v) for k, v in VEHICLE_WEIGHTS.items()},
            focal_length_px=None,
        )

        now = time.time()
        avg_speed_kph = None
        if prev_detections is not None and prev_time is not None:
            dt_s = max(0.001, now - prev_time)
            avg_speed_mps = estimate_avg_speed_mps(
                detections,
                prev_detections,
                dt_s,
                deadband_m=0.5,
                dt_min_s=0.1,
                dt_max_s=2.0,
            )
            if avg_speed_mps is not None:
                avg_speed_kph = float(avg_speed_mps * 3.6)

        prev_detections = detections
        prev_time = now

        density = float(vehicle_data.get("density", 0.0) or 0.0)
        load_tons = float(vehicle_data.get("load_tons", 0.0) or 0.0)
        rho_max_run = float(np.random.uniform(0.18, 0.22)) if randomize_constants else float(rho_max)
        road_length_run = float(bridge_config.total_length_m) * (float(np.random.uniform(0.98, 1.02)) if randomize_constants else 1.0)
        k_mpa_per_ton_run = float(np.random.uniform(0.25, 0.35)) if randomize_constants else float(k_mpa_per_ton)
        stress_mpa_proxy = float(load_tons_to_stress_mpa(load_tons, k_mpa_per_ton_run))
        v_max_mps = float(((avg_speed_kph if avg_speed_kph is not None else v_max_kmh_default)) / 3.6)

        sim = run_lwr_simulation(
            initial_density=density,
            road_length_m=road_length_run,
            v_max_mps=v_max_mps,
            inject_jam=False,
            rho_max=rho_max_run,
        )

        env_breakdown = calculate_environmental_stress(
            temperature=env_temperature,
            humidity=env_humidity,
            precipitation=env_precipitation,
            freeze_thaw_cycles=int(env_freeze_thaw_7day),
            wind_speed=env_wind_speed,
            bridge_age=bridge_config.age_years,
        )
        env_stress = float(env_breakdown.get("combined", 0.0))
        if randomize_constants:
            env_stress = float(np.clip(np.random.normal(env_stress, 2.0), 0.0, 100.0))

        sigma_S = max(1e-6, abs(stress_mpa_proxy) * sigma_S_ratio)
        reliability_beta = float(
            compute_reliability_index(
                mu_R=float(mu_R),
                sigma_R=float(sigma_R),
                mu_S=float(stress_mpa_proxy),
                sigma_S=float(sigma_S),
            )
        )

        out_row = {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "density": density,
            "speed_kph": avg_speed_kph,
            "v_max_kph_used": v_max_mps * 3.6,
            "load_tons": load_tons,
            "stress_mpa_proxy": stress_mpa_proxy,
            "reliability_beta": reliability_beta,
            "env_stress": env_stress,
            "k_mpa_per_ton": k_mpa_per_ton_run,
            "sim_fatigue": sim.get("fatigue"),
            "sim_shockwave_speed": sim.get("shockwave_speed"),
            "avg_density": sim.get("avg_density"),
            "max_stress": sim.get("max_stress"),
            "rho_max": rho_max_run,
            "road_length_m": road_length_run,
            "source": "live_camera",
        }
        _write_row(output_csv, out_row)
        time.sleep(interval_sec)


# -----------------------------
# Plotting (Plotly optional)
# -----------------------------

def try_plotly():
    try:
        import plotly.express as px  # noqa: F401
        import plotly.graph_objects as go  # noqa: F401
        return True
    except Exception:
        return False


def save_plotly_fig(fig, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(outpath), width=1920, height=1080, scale=4)
    except Exception as exc:
        html_path = outpath.with_suffix(".html")
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        print(f"[warn] PNG export failed ({exc}); wrote HTML instead: {html_path}")


def make_figs(df: pd.DataFrame, outdir: Path) -> List[Path]:
    """
    Create a small set of standard figures from experiment outputs.
    Saves HTML so you can open them in a browser.
    """
    if not try_plotly():
        return []

    import plotly.express as px
    import plotly.graph_objects as go

    line_colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    discrete_colors = ["#1b9e77", "#d95f02", "#7570b3", "#66a61e", "#e6ab02"]
    heatmap_scale = "Viridis"
    template = "plotly_white"
    base_font = dict(size=28, family="Times New Roman", color="#000000")
    axis_font = dict(size=32, family="Times New Roman", color="#000000")
    title_font = dict(size=48, family="Times New Roman", color="#000000")
    legend_font = dict(size=32, family="Times New Roman", color="#000000")

    figs: List[Path] = []

    # 1) Density sweep line charts
    d = df[df["sweep"].eq("density")].copy()
    if len(d):
        fig = px.line(
            d, x="density", y="fatigue_mean", markers=True,
            title="Fatigue vs Density (mean)",
            color_discrete_sequence=discrete_colors,
        )
        fig.update_layout(
            template=template,
            font=base_font,
            title_font=title_font,
            legend_font=legend_font,
            height=900,
            margin=dict(l=100, r=60, t=120, b=180),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(title_font=axis_font, tickfont=axis_font)
        fig.update_yaxes(title_font=axis_font, tickfont=axis_font)
        p = outdir / "figs" / "fatigue_vs_density.png"
        save_plotly_fig(fig, p); figs.append(p)

        fig = px.line(
            d, x="density", y="shockwave_mean", markers=True,
            title="Shockwave speed vs Density (mean)",
            color_discrete_sequence=discrete_colors,
        )
        fig.update_layout(
            template=template,
            font=base_font,
            title_font=title_font,
            legend_font=legend_font,
            height=900,
            margin=dict(l=100, r=60, t=120, b=180),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(title_font=axis_font, tickfont=axis_font)
        fig.update_yaxes(title_font=axis_font, tickfont=axis_font)
        p = outdir / "figs" / "shockwave_vs_density.png"
        save_plotly_fig(fig, p); figs.append(p)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["density"], y=d["fatigue_p90"], mode="lines", name="p90", line=dict(width=0)))
        fig.add_trace(go.Scatter(x=d["density"], y=d["fatigue_p50"], mode="lines", name="p50", line=dict(color=line_colors[0])))
        fig.add_trace(go.Scatter(
            x=d["density"], y=d["fatigue_p90"], mode="lines",
            fill="tonexty", line=dict(width=0), name="p90 band", showlegend=False, fillcolor="rgba(27,158,119,0.2)"
        ))
        fig.update_layout(
            title="Fatigue vs Density (p50/p90 band)",
            xaxis_title="density",
            yaxis_title="fatigue",
            template=template,
            font=base_font,
            title_font=title_font,
            legend_font=legend_font,
            height=900,
            margin=dict(l=100, r=60, t=120, b=180),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(title_font=axis_font, tickfont=axis_font)
        fig.update_yaxes(title_font=axis_font, tickfont=axis_font)
        p = outdir / "figs" / "fatigue_vs_density_bands.png"
        save_plotly_fig(fig, p); figs.append(p)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["density"], y=d["shockwave_p90"], mode="lines", name="p90", line=dict(width=0)))
        fig.add_trace(go.Scatter(x=d["density"], y=d["shockwave_p50"], mode="lines", name="p50", line=dict(color=line_colors[1])))
        fig.add_trace(go.Scatter(
            x=d["density"], y=d["shockwave_p90"], mode="lines",
            fill="tonexty", line=dict(width=0), name="p90 band", showlegend=False, fillcolor="rgba(217,95,2,0.2)"
        ))
        fig.update_layout(
            title="Shockwave vs Density (p50/p90 band)",
            xaxis_title="density",
            yaxis_title="shockwave",
            template=template,
            font=base_font,
            title_font=title_font,
            legend_font=legend_font,
            height=900,
            margin=dict(l=100, r=60, t=120, b=180),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(title_font=axis_font, tickfont=axis_font)
        fig.update_yaxes(title_font=axis_font, tickfont=axis_font)
        p = outdir / "figs" / "shockwave_vs_density_bands.png"
        save_plotly_fig(fig, p); figs.append(p)

    # 2) v_max sweep
    v = df[df["sweep"].eq("v_max")].copy()
    if len(v):
        fig = px.line(
            v, x="v_max_kmh", y="fatigue_mean", markers=True,
            title="Fatigue vs v_max (mean)",
            color_discrete_sequence=discrete_colors,
        )
        fig.update_layout(
            template=template,
            font=base_font,
            title_font=title_font,
            legend_font=legend_font,
            height=900,
            margin=dict(l=100, r=60, t=120, b=180),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(title_font=axis_font, tickfont=axis_font)
        fig.update_yaxes(title_font=axis_font, tickfont=axis_font)
        p = outdir / "figs" / "fatigue_vs_vmax.png"
        save_plotly_fig(fig, p); figs.append(p)

        fig = px.line(
            v, x="v_max_kmh", y="shockwave_mean", markers=True,
            title="Shockwave speed vs v_max (mean)",
            color_discrete_sequence=discrete_colors,
        )
        fig.update_layout(
            template=template,
            font=base_font,
            title_font=title_font,
            legend_font=legend_font,
            height=900,
            margin=dict(l=100, r=60, t=120, b=180),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(title_font=axis_font, tickfont=axis_font)
        fig.update_yaxes(title_font=axis_font, tickfont=axis_font)
        p = outdir / "figs" / "shockwave_vs_vmax.png"
        save_plotly_fig(fig, p); figs.append(p)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=v["v_max_kmh"], y=v["fatigue_p90"], mode="lines", name="p90", line=dict(width=0)))
        fig.add_trace(go.Scatter(x=v["v_max_kmh"], y=v["fatigue_p50"], mode="lines", name="p50"))
        fig.add_trace(go.Scatter(
            x=v["v_max_kmh"], y=v["fatigue_p90"], mode="lines",
            fill="tonexty", line=dict(width=0), name="p90 band", showlegend=False
        ))
        fig.update_layout(
            title="Fatigue vs v_max (p50/p90 band)",
            xaxis_title="v_max_kmh",
            yaxis_title="fatigue",
            template=template,
            font=base_font,
            title_font=title_font,
            legend_font=legend_font,
            height=900,
            margin=dict(l=100, r=60, t=120, b=180),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(title_font=axis_font, tickfont=axis_font)
        fig.update_yaxes(title_font=axis_font, tickfont=axis_font)
        p = outdir / "figs" / "fatigue_vs_vmax_bands.png"
        save_plotly_fig(fig, p); figs.append(p)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=v["v_max_kmh"], y=v["shockwave_p90"], mode="lines", name="p90", line=dict(width=0)))
        fig.add_trace(go.Scatter(x=v["v_max_kmh"], y=v["shockwave_p50"], mode="lines", name="p50"))
        fig.add_trace(go.Scatter(
            x=v["v_max_kmh"], y=v["shockwave_p90"], mode="lines",
            fill="tonexty", line=dict(width=0), name="p90 band", showlegend=False
        ))
        fig.update_layout(
            title="Shockwave vs v_max (p50/p90 band)",
            xaxis_title="v_max_kmh",
            yaxis_title="shockwave",
            template=template,
            font=base_font,
            title_font=title_font,
            legend_font=legend_font,
            height=900,
            margin=dict(l=100, r=60, t=120, b=180),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(title_font=axis_font, tickfont=axis_font)
        fig.update_yaxes(title_font=axis_font, tickfont=axis_font)
        p = outdir / "figs" / "shockwave_vs_vmax_bands.png"
        save_plotly_fig(fig, p); figs.append(p)

    # 3) Jam probability sweep
    j = df[df["sweep"].eq("jam_probability")].copy()
    if len(j):
        fig = px.line(
            j, x="inject_jam_probability", y="fatigue_mean", markers=True,
            title="Fatigue vs Jam Probability (mean)",
            color_discrete_sequence=discrete_colors,
        )
        fig.update_layout(
            template=template,
            font=base_font,
            title_font=title_font,
            legend_font=legend_font,
            height=900,
            margin=dict(l=100, r=60, t=120, b=180),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(title_font=axis_font, tickfont=axis_font)
        fig.update_yaxes(title_font=axis_font, tickfont=axis_font)
        p = outdir / "figs" / "fatigue_vs_jamprob.png"
        save_plotly_fig(fig, p); figs.append(p)

        fig = px.line(
            j, x="inject_jam_probability", y="jam_rate", markers=True,
            title="Jam Rate vs Jam Probability",
            color_discrete_sequence=discrete_colors,
        )
        fig.update_layout(
            template=template,
            font=base_font,
            title_font=title_font,
            legend_font=legend_font,
            height=900,
            margin=dict(l=100, r=60, t=120, b=180),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(title_font=axis_font, tickfont=axis_font)
        fig.update_yaxes(title_font=axis_font, tickfont=axis_font)
        p = outdir / "figs" / "jamrate_vs_jamprob.png"
        save_plotly_fig(fig, p); figs.append(p)

    # 4) Grid heatmap (fatigue_mean)
    g = df[df["sweep"].eq("grid")].copy()
    if len(g):
        pivot = g.pivot(index="density", columns="v_max_kmh", values="fatigue_mean")
        fig = px.imshow(
            pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            labels={"x": "v_max_kmh", "y": "density", "color": "fatigue_mean"},
            title="Fatigue Mean Heatmap (density x v_max)",
            aspect="auto",
            color_continuous_scale=heatmap_scale,
        )
        fig.update_layout(
            template=template,
            font=base_font,
            title_font=title_font,
            legend_font=legend_font,
            height=900,
            margin=dict(l=100, r=60, t=120, b=180),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(title_font=axis_font, tickfont=axis_font)
        fig.update_yaxes(title_font=axis_font, tickfont=axis_font)
        p = outdir / "figs" / "fatigue_heatmap_density_vmax.png"
        save_plotly_fig(fig, p); figs.append(p)

        pivot = g.pivot(index="density", columns="v_max_kmh", values="shockwave_mean")
        fig = px.imshow(
            pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            labels={"x": "v_max_kmh", "y": "density", "color": "shockwave_mean"},
            title="Shockwave Mean Heatmap (density x v_max)",
            aspect="auto",
            color_continuous_scale=heatmap_scale,
        )
        fig.update_layout(
            template=template,
            font=base_font,
            title_font=title_font,
            legend_font=legend_font,
            height=900,
            margin=dict(l=100, r=60, t=120, b=180),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(title_font=axis_font, tickfont=axis_font)
        fig.update_yaxes(title_font=axis_font, tickfont=axis_font)
        p = outdir / "figs" / "shockwave_heatmap_density_vmax.png"
        save_plotly_fig(fig, p); figs.append(p)

        pivot = g.pivot(index="density", columns="v_max_kmh", values="fatigue_p90")
        fig = px.imshow(
            pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            labels={"x": "v_max_kmh", "y": "density", "color": "fatigue_p90"},
            title="Fatigue P90 Heatmap (density x v_max)",
            aspect="auto",
            color_continuous_scale=heatmap_scale,
        )
        fig.update_layout(
            template=template,
            font=base_font,
            title_font=title_font,
            legend_font=legend_font,
            height=900,
            margin=dict(l=100, r=60, t=120, b=180),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(title_font=axis_font, tickfont=axis_font)
        fig.update_yaxes(title_font=axis_font, tickfont=axis_font)
        p = outdir / "figs" / "fatigue_p90_heatmap_density_vmax.png"
        save_plotly_fig(fig, p); figs.append(p)

    # 5) Monte Carlo distributions
    mc = df[df["sweep"].eq("monte_carlo")].copy()
    if len(mc):
        fig = px.histogram(mc, x="fatigue", nbins=40, title="Monte Carlo Fatigue Distribution")
        fig.update_traces(marker_color=discrete_colors[0])
        fig.update_layout(
            template=template,
            font=base_font,
            title_font=title_font,
            legend_font=legend_font,
            height=900,
            margin=dict(l=100, r=60, t=120, b=180),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(title_font=axis_font, tickfont=axis_font)
        fig.update_yaxes(title_font=axis_font, tickfont=axis_font)
        p = outdir / "figs" / "mc_fatigue_hist.png"
        save_plotly_fig(fig, p); figs.append(p)

        fig = px.scatter(
            mc, x="density", y="fatigue", color="jam_injected",
            title="MC Scatter: Fatigue vs Density",
            color_discrete_sequence=discrete_colors,
        )
        fig.update_layout(
            template=template,
            font=base_font,
            title_font=title_font,
            legend_font=legend_font,
            height=900,
            margin=dict(l=100, r=60, t=120, b=180),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.35, x=0.5, xanchor="center"),
        )
        fig.update_xaxes(title_font=axis_font, tickfont=axis_font)
        fig.update_yaxes(title_font=axis_font, tickfont=axis_font)
        p = outdir / "figs" / "mc_scatter_fatigue_density.png"
        save_plotly_fig(fig, p); figs.append(p)

        fig = px.scatter(
            mc, x="v_max", y="fatigue", color="jam_injected",
            title="MC Scatter: Fatigue vs v_max",
            color_discrete_sequence=discrete_colors,
        )
        fig.update_layout(
            template=template,
            font=base_font,
            title_font=title_font,
            legend_font=legend_font,
            height=900,
            margin=dict(l=100, r=60, t=120, b=180),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.35, x=0.5, xanchor="center"),
        )
        fig.update_xaxes(title_font=axis_font, tickfont=axis_font)
        fig.update_yaxes(title_font=axis_font, tickfont=axis_font)
        p = outdir / "figs" / "mc_scatter_fatigue_vmax.png"
        save_plotly_fig(fig, p); figs.append(p)

        fig = px.histogram(mc, x="shockwave_speed", nbins=40, title="Monte Carlo Shockwave Distribution")
        fig.update_traces(marker_color=discrete_colors[1])
        fig.update_layout(
            template=template,
            font=base_font,
            title_font=title_font,
            legend_font=legend_font,
            height=900,
            margin=dict(l=100, r=60, t=120, b=180),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(title_font=axis_font, tickfont=axis_font)
        fig.update_yaxes(title_font=axis_font, tickfont=axis_font)
        p = outdir / "figs" / "mc_shockwave_hist.png"
        save_plotly_fig(fig, p); figs.append(p)

        fig = px.density_heatmap(
            mc, x="density", y="shockwave_speed",
            nbinsx=30, nbinsy=30, title="MC Density vs Shockwave (density heatmap)",
            color_continuous_scale=heatmap_scale,
        )
        fig.update_layout(
            template=template,
            font=base_font,
            title_font=title_font,
            legend_font=legend_font,
            height=900,
            margin=dict(l=100, r=60, t=120, b=180),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(title_font=axis_font, tickfont=axis_font)
        fig.update_yaxes(title_font=axis_font, tickfont=axis_font)
        p = outdir / "figs" / "mc_density_shockwave_heatmap.png"
        save_plotly_fig(fig, p); figs.append(p)

    return figs


# -----------------------------
# Main runner
# -----------------------------

def run_all(outdir: Path, spec: SweepSpec, mc_runs: int = 500, live_density: Optional[float] = None) -> Dict[str, str]:
    _ensure_outdir(outdir)
    _set_seed(spec.seed)

    meta_path = outdir / "tables" / "spec.json"
    meta_path.write_text(json.dumps(asdict(spec), indent=2))

    df_all = pd.concat(
        [
            sweep_density(spec),
            sweep_vmax(spec),
            sweep_jam_probability(spec),
            grid_heatmap(spec),
            monte_carlo_bundle(spec, num_runs=mc_runs, live_density=live_density),
        ],
        ignore_index=True,
    )

    csv_path = outdir / "tables" / "experiments_all.csv"
    df_all.to_csv(csv_path, index=False)

    fig_paths = make_figs(df_all, outdir)

    return {
        "outdir": str(outdir),
        "csv": str(csv_path),
        "spec": str(meta_path),
        "figs": json.dumps([str(p) for p in fig_paths], indent=2),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run batch experiments and export CSV + figures.")
    p.add_argument("--mode", type=str, default="synthetic",
                   choices=["synthetic", "live_csv", "live_camera"],
                   help="Run mode: synthetic sweeps or live data source")
    p.add_argument("--outdir", type=str, default="experiments_out", help="Output directory")
    p.add_argument("--name", type=str, default="peace_bridge_batch", help="Experiment name")
    p.add_argument("--road_length_m", type=float, default=1768.0, help="Road length (m)")
    p.add_argument("--rho_max", type=float, default=0.20, help="Jam density (vehicles/m)")
    p.add_argument("--jam_prob", type=float, default=0.30, help="Jam injection probability")
    p.add_argument("--reps", type=int, default=50, help="Repetitions per sweep point")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--mc_runs", type=int, default=500, help="Monte Carlo run count")
    p.add_argument("--live_density", type=float, default=None, help="If set, MC samples around this density")
    p.add_argument("--duration_sec", type=float, default=600, help="Live mode duration (seconds)")
    p.add_argument("--interval_sec", type=float, default=0.25, help="Live mode sample interval (seconds)")
    p.add_argument("--output_csv", type=str, default="experiments_out/live_results.csv", help="Live mode output CSV")
    p.add_argument("--live_csv_path", type=str, default="", help="Input CSV path for live_csv mode")
    p.add_argument("--density_col", type=str, default="density", help="Column name for density in live CSV")
    p.add_argument("--speed_col", type=str, default="", help="Column name for speed (kph) in live CSV")
    p.add_argument("--load_col", type=str, default="", help="Column name for load (tons) in live CSV")
    p.add_argument("--env_temp_col", type=str, default="", help="Column name for temperature in live CSV")
    p.add_argument("--env_humidity_col", type=str, default="", help="Column name for humidity in live CSV")
    p.add_argument("--env_precip_col", type=str, default="", help="Column name for precipitation in live CSV")
    p.add_argument("--env_freeze_thaw_col", type=str, default="", help="Column name for freeze-thaw cycles in live CSV")
    p.add_argument("--env_wind_col", type=str, default="", help="Column name for wind speed in live CSV")
    p.add_argument("--camera_name", type=str, default="Peace Bridge - Canada Bound", help="Camera name for live_camera")
    p.add_argument("--confidence", type=float, default=0.15, help="YOLO confidence for live_camera")
    p.add_argument("--lane_divider", type=float, default=0.43, help="Lane divider for live_camera")
    p.add_argument("--use_roi", action="store_true", help="Enable ROI for live_camera")
    p.add_argument("--vmax_kph_default", type=float, default=80.0, help="Fallback v_max when speed unavailable")
    p.add_argument("--k_mpa_per_ton", type=float, default=0.6, help="Stress proxy coefficient")
    p.add_argument("--env_temperature", type=float, default=10.0, help="Fallback temperature (C)")
    p.add_argument("--env_humidity", type=float, default=50.0, help="Fallback humidity (%)")
    p.add_argument("--env_precipitation", type=float, default=0.0, help="Fallback precipitation (mm)")
    p.add_argument("--env_freeze_thaw_7day", type=float, default=0.0, help="Fallback freeze-thaw cycles (7d)")
    p.add_argument("--env_wind_speed", type=float, default=5.0, help="Fallback wind speed (m/s)")
    p.add_argument("--bridge_age_years", type=float, default=99.0, help="Bridge age (years)")
    p.add_argument("--mu_R", type=float, default=250.0, help="Resistance mean (MPa)")
    p.add_argument("--sigma_R", type=float, default=25.0, help="Resistance std dev (MPa)")
    p.add_argument("--sigma_S_ratio", type=float, default=0.10, help="Load std dev ratio")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "synthetic":
        outdir = Path(args.outdir).resolve()
        spec = SweepSpec(
            name=args.name,
            road_length_m=float(args.road_length_m),
            rho_max=float(args.rho_max),
            inject_jam_probability=float(args.jam_prob),
            n_reps=int(args.reps),
            seed=int(args.seed),
        )
        summary = run_all(outdir=outdir, spec=spec, mc_runs=int(args.mc_runs), live_density=args.live_density)
        print("\n=== Experiment run complete ===")
        for k, v in summary.items():
            print(f"{k}: {v}")
        return

    output_csv = Path(args.output_csv).resolve()
    if args.mode == "live_csv":
        if not args.live_csv_path:
            raise SystemExit("--live_csv_path is required for live_csv mode")
        run_live_csv(
            input_csv=Path(args.live_csv_path).resolve(),
            output_csv=output_csv,
            duration_sec=float(args.duration_sec),
            interval_sec=float(args.interval_sec),
            density_col=str(args.density_col),
            speed_col=str(args.speed_col) if args.speed_col else None,
            load_col=str(args.load_col) if args.load_col else None,
            env_temp_col=str(args.env_temp_col) if args.env_temp_col else None,
            env_humidity_col=str(args.env_humidity_col) if args.env_humidity_col else None,
            env_precip_col=str(args.env_precip_col) if args.env_precip_col else None,
            env_freeze_thaw_col=str(args.env_freeze_thaw_col) if args.env_freeze_thaw_col else None,
            env_wind_col=str(args.env_wind_col) if args.env_wind_col else None,
            v_max_kmh_default=float(args.vmax_kph_default),
            road_length_m=float(args.road_length_m),
            rho_max=float(args.rho_max),
            k_mpa_per_ton=float(args.k_mpa_per_ton),
            env_temperature=float(args.env_temperature),
            env_humidity=float(args.env_humidity),
            env_precipitation=float(args.env_precipitation),
            env_freeze_thaw_7day=float(args.env_freeze_thaw_7day),
            env_wind_speed=float(args.env_wind_speed),
            bridge_age_years=float(args.bridge_age_years),
            mu_R=float(args.mu_R),
            sigma_R=float(args.sigma_R),
            sigma_S_ratio=float(args.sigma_S_ratio),
        )
        print(f"\n=== Live CSV run complete ===\nSaved: {output_csv}")
        return

    if args.mode == "live_camera":
        if args.camera_name not in CAMERAS:
            raise SystemExit(f"Unknown camera_name. Options: {list(CAMERAS.keys())}")
        run_live_camera(
            output_csv=output_csv,
            duration_sec=float(args.duration_sec),
            interval_sec=float(args.interval_sec),
            camera_name=str(args.camera_name),
            confidence=float(args.confidence),
            lane_divider=float(args.lane_divider),
            use_roi=bool(args.use_roi),
            v_max_kmh_default=float(args.vmax_kph_default),
            rho_max=float(args.rho_max),
            k_mpa_per_ton=float(args.k_mpa_per_ton),
            env_temperature=float(args.env_temperature),
            env_humidity=float(args.env_humidity),
            env_precipitation=float(args.env_precipitation),
            env_freeze_thaw_7day=float(args.env_freeze_thaw_7day),
            env_wind_speed=float(args.env_wind_speed),
            mu_R=float(args.mu_R),
            sigma_R=float(args.sigma_R),
            sigma_S_ratio=float(args.sigma_S_ratio),
        )
        print(f"\n=== Live camera run complete ===\nSaved: {output_csv}")
        return


if __name__ == "__main__":
    main()
