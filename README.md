# Weather Prediction Bot — Intelligent, Multi-Source, Probabilistic

A modular Python system that synthesizes numerical forecasts, historical
patterns, news signals, geographic context, and satellite data into
calibrated probabilistic weather predictions with natural-language
explanations and a continuous-learning feedback loop.

---

## 1. System Architecture

```
                       ┌──────────────────────────────────────┐
                       │           Client / Dashboard          │
                       │  (Streamlit UI, REST, WebSocket)      │
                       └───────────────┬──────────────────────┘
                                       │
                       ┌───────────────▼──────────────────────┐
                       │           FastAPI Service             │
                       │   /predict  /explain  /feedback       │
                       └───────────────┬──────────────────────┘
                                       │
          ┌────────────────────────────┼───────────────────────────────┐
          │                            │                               │
┌─────────▼────────┐       ┌───────────▼──────────┐       ┌────────────▼──────────┐
│ Ingestion Layer  │       │  Prediction Engine   │       │  Feedback & Retrain   │
│                  │       │                      │       │                       │
│ • Weather APIs   │──────▶│ • Short-term (LGBM   │──────▶│ • SQLite / Postgres   │
│ • News / RSS     │       │   quantile regress.) │       │ • Skill scoring       │
│ • Historical DB  │       │ • Mid-term (Prophet  │       │ • Source reweighting  │
│ • Geographic     │       │   / LSTM)            │       │ • Periodic retrain    │
│ • Satellite/Radar│       │ • Ensemble stack     │       │                       │
│ • NLP signals    │       │ • Anomaly detector   │       │                       │
└──────────────────┘       └──────────┬───────────┘       └───────────────────────┘
                                      │
                       ┌──────────────▼───────────────┐
                       │    Explainer + Alert Engine   │
                       │  (SHAP + NLG + thresholds)    │
                       └──────────────────────────────┘
```

### Data flow (one prediction cycle)

1. **Ingestion** — N parallel adapters pull current obs + forecast grids, news articles, historical climate normals, geographic context, and radar frames. Each returns a normalized `SourceReading` dataclass with a `reliability` hint.
2. **Processing** — Feature engineering builds the model input vector: lag features, rolling statistics, seasonal Fourier terms, geographic features (elevation, coast distance, UHI index), and NLP-derived event signals (e.g. `hurricane_signal=0.8`).
3. **Prediction** — Short-term (LightGBM quantile regression for 0.1/0.5/0.9 quantiles) and mid-term (Prophet with extra regressors, optional LSTM) produce point + interval forecasts per variable (temp, precip, wind). A stacking meta-learner combines them with dynamic per-source weights learned from skill history.
4. **Anomaly check** — Isolation Forest over the feature vector + CUSUM on residual streams flag unusual conditions.
5. **Explain** — SHAP attributions are mapped to plain-English templates ("strong pressure drop over the next 12h + incoming cold front drove the rain probability up").
6. **Alerts** — Threshold + anomaly combined rules fire alerts (extreme heat, severe storm, flash flood risk).
7. **Persist** — Prediction, features, and confidence stored. When actuals arrive, errors are computed and source weights + model skill scores are updated.

---

## 2. Repository Layout

```
weather_bot/
├── README.md                     ← this file
├── requirements.txt
├── config/
│   └── config.yaml               ← API keys, weights, thresholds
├── src/
│   ├── ingestion/
│   │   ├── base.py               ← SourceReading dataclass, base adapter
│   │   ├── weather_api.py        ← Open-Meteo + OpenWeatherMap adapters
│   │   ├── news_scraper.py       ← RSS + NewsAPI async fetcher
│   │   ├── historical.py         ← Meteostat climate normals
│   │   ├── geographic.py         ← Elevation, coast distance, UHI
│   │   └── satellite.py          ← RainViewer radar tiles
│   ├── processing/
│   │   ├── features.py           ← Feature engineering pipeline
│   │   └── nlp.py                ← News → weather event signals
│   ├── models/
│   │   ├── short_term.py         ← LGBM quantile regressor
│   │   ├── mid_term.py           ← Prophet-based multi-day
│   │   ├── ensemble.py           ← Stacking + dynamic weights
│   │   └── anomaly.py            ← Isolation Forest + CUSUM
│   ├── feedback/
│   │   ├── store.py              ← SQLAlchemy models
│   │   └── evaluator.py          ← Skill scoring, weight updates
│   ├── explain/
│   │   └── explainer.py          ← SHAP → NLG
│   ├── alerts/
│   │   └── alert_engine.py       ← Alert rules
│   ├── api/
│   │   └── app.py                ← FastAPI service
│   └── orchestrator.py           ← Runs one full prediction cycle
├── examples/
│   └── run_cycle.py              ← End-to-end example
└── tests/
```

---

## 3. Key Algorithms & Why

| Component          | Algorithm                                | Why                                                                 |
|--------------------|------------------------------------------|---------------------------------------------------------------------|
| Short-term (0–72h) | LightGBM **quantile regression**         | Non-parametric, handles mixed features, native prediction intervals |
| Mid-term (3–7d)    | Prophet with extra regressors (or LSTM)  | Captures seasonality + holidays + external signals, robust          |
| Ensemble           | Stacked meta-learner (ridge) + per-source weights | Exploits strengths of each model; weights adapt to recent skill |
| Confidence         | Quantile spread + Bayesian calibration   | Interval width → % confidence after isotonic calibration            |
| Anomaly            | Isolation Forest + CUSUM on residuals    | Catches both point anomalies and drifting regimes                   |
| NLP signals        | spaCy NER + keyword severity + (optional) transformer classifier | Converts free-text news into numeric signals |
| Explainability     | SHAP TreeExplainer + NLG templates       | Feature attributions mapped to human phrases                        |
| Feedback           | EMA of skill scores per source/model     | Dynamic reweighting without full retrain                            |

### Confidence score

For each predicted variable *y*:

```
interval_width = q90 - q10
normalized     = interval_width / climatology_std
confidence_%   = 100 * sigmoid(-k * (normalized - 1))   # calibrated via isotonic regression
```

Sources that have been reliable *for this location and season recently* get upweighted in the ensemble, computed as:

```
w_s  =  exp(-λ · EMA(|error_s|))  /  Σ_s  exp(-λ · EMA(|error_s|))
```

---

## 4. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config/config.example.yaml config/config.yaml   # fill in API keys
python -m src.api.app                               # REST on :8000
# or run a one-shot cycle:
python examples/run_cycle.py --location "Madrid,ES"
```

---

## 5. Scaling & Accuracy Improvements

- **Horizontal scale**: split ingestion into Celery workers; each source is a task. Redis as broker. Cache normalized readings for 5–15 min per (lat, lon, source).
- **Data lake**: append-only Parquet in S3/MinIO partitioned by `(yyyymmdd, source)`. Feature store (Feast) for training/serving consistency.
- **Better short-term models**: replace LightGBM with **Temporal Fusion Transformer** or **N-BEATS** once you have 1–2 years of labeled per-location data.
- **Nowcasting**: for 0–2h precipitation, use optical-flow or a small ConvLSTM over radar tiles — dramatically better than any NWP.
- **Bias correction**: learn a per-location residual model (GBM) that corrects the NWP forecast using recent local obs. Classic MOS (Model Output Statistics) approach; typically cuts RMSE 10–30%.
- **Probabilistic calibration**: evaluate with **CRPS** (Continuous Ranked Probability Score), not just RMSE. Apply isotonic regression on held-out predictions to calibrate confidence.
- **Ensemble of NWPs**: pull from multiple forecast providers (ECMWF, GFS, ICON via Open-Meteo's `models=` parameter) and learn weights per lead time and region.
- **Active retraining**: trigger retrain when rolling CRPS degrades by > X% vs. baseline, not on a fixed schedule.
- **Human feedback loop**: let users flag bad predictions; use as hard-negative mining signal.
- **Monitoring**: Prometheus + Grafana on ingestion latency, source availability, per-variable MAE by lead time, alert precision/recall.
