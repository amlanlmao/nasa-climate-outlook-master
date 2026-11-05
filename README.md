# NASA Climate Outlook Dashboard

A Streamlit dashboard that downloads real NASA GISTEMP temperature observations, forecasts the
next 6–24 months, and reports rolling-origin mean absolute error (MAE).

## Run

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app fetches the latest data from NASA at runtime and caches it for one hour. If the
NASA GISS host is unavailable, it quickly falls back to NOAA PSL's mirror of a NASA GISTEMP series.

## Scope

The initial version intentionally predicts one well-defined climate indicator: global surface
temperature anomaly. The observations are real; the forecast is a transparent experimental
baseline using seasonal averages and a recent linear trend, not a claim that short-term climate
variability can be predicted exactly. The dashboard compares it with a persistence baseline and
uses rolling-origin validation, but it is not yet a production climate model.

The direct source is NASA GISTEMP v4. If NASA's GISS host is unavailable, the app uses NOAA PSL's
mirror of a NASA temperature series. The fallback may have a different land/land-ocean coverage,
which is shown in the dashboard's data-source label.

Before production use, add additional NASA datasets, stronger statistical or machine-learning
models, uncertainty calibration, data versioning, scheduled ingestion, and evaluation on a
strictly held-out period.

Source: [NASA GISS Surface Temperature Analysis](https://data.giss.nasa.gov/gistemp/).
