## Hybrid Digital Twin for Bridge Traffic, Fatigue, and Reliability Monitoring

This repository implements a Hybrid Digital Twin framework for bridge infrastructure monitoring by combining real-time traffic observations, physics-based traffic flow modeling, environmental degradation analysis, stochastic simulation, and machine-learning–based fatigue prediction.

The system is designed as a research and proof-of-concept platform demonstrating how low-cost public data sources and models can be integrated to support predictive maintenance and infrastructure resilience analysis.

---

## Project Overview

The digital twin integrates four complementary layers:

1. Sensing Layer
   - Live traffic camera feeds (YouTube / HLS)
   - Computer vision–based vehicle detection
   - Real-time weather data from Open-Meteo

2. Physics-Based Modeling Layer
   - Macroscopic traffic flow using the Lighthill–Whitham–Richards (LWR) model
   - Shockwave propagation and congestion modeling
   - Traffic-induced stress estimation

3. Stochastic and Reliability Layer
   - Monte Carlo simulation of traffic scenarios
   - Fatigue accumulation using Miner’s Rule
   - Structural reliability index (β) computation

4. Data-Driven Layer
   - Random Forest regression for fatigue prediction
   - Training on combined historical weather data and simulated traffic scenarios
   - Feature importance and sensitivity analysis

The system outputs a normalized fatigue score (0–100) and a reliability index, enabling scenario-based assessment of bridge health under varying traffic and environmental conditions.

---

## Repository Structure

.
├── Peace_Bridge.py
├── bridge_modified.py
└── README.md

### bridge_modified.py – Core Digital Twin Engine

This file contains the analytical and computational core of the project.

Key responsibilities include:
- LWR traffic flow simulation and shockwave estimation
- Traffic stress and congestion modeling
- Environmental stress modeling (freeze–thaw, precipitation, temperature, wind, aging)
- Fatigue damage computation using Miner’s Rule
- Reliability index (β) calculation
- Monte Carlo simulation for uncertainty and sensitivity analysis
- Machine learning training and inference using Random Forest models
- Integration of historical weather data for data-driven fatigue learning

This module is independent of visualization and can be reused in other applications or extended for different bridge assets.

---

### Peace_Bridge.py – Interactive Monitoring and Visualization

This file implements the Streamlit-based user interface and live monitoring workflow.

Key features include:
- Live camera ingestion and frame capture
- YOLO-based vehicle detection and classification
- Region-of-interest (ROI) traffic counting and load estimation
- Real-time weather retrieval
- Monitoring sessions with periodic captures
- Fatigue damage and reliability tracking over time
- Interactive scenario analysis (traffic growth, truck percentage, environmental changes)
- Exportable CSV logs and HTML reports (with images)
- Validation report template for comparison with official traffic counts

This file orchestrates the full digital twin pipeline and visualizes outputs from bridge_modified.py.

---

## Methodology Summary

- Traffic Modeling: Macroscopic LWR model with probabilistic jam injection
- Fatigue Modeling: Cumulative stress integration and Miner’s Rule
- Environmental Effects: Weather-driven corrosion and degradation heuristics
- Uncertainty Handling: Monte Carlo simulation
- Machine Learning: Random Forest regression trained on synthetic–historical hybrid datasets
- Reliability Analysis: Structural reliability index based on stress statistics

---

## Dependencies

Python libraries:
- streamlit
- ultralytics
- opencv-python
- numpy
- pandas
- scikit-learn
- plotly
- scipy
- requests
- yt-dlp
- Pillow

System dependencies:
- ffmpeg

---

## Running the Application

To launch the interactive dashboard:

streamlit run Peace_Bridge.py

The application will be available at:
http://localhost:8501

---

## Use Cases

- Digital twin architecture prototyping
- Infrastructure fatigue and reliability research
- Traffic-induced degradation analysis
- Predictive maintenance scenario exploration
- Academic demonstrations in modeling and simulation

---

## Limitations

- Vehicle weights and loads are estimated averages
- Camera-based detection introduces uncertainty
- No direct integration with structural sensors
- Fatigue score is a proxy metric, not a certified inspection measure

This system is intended for research and educational purposes only.

---
