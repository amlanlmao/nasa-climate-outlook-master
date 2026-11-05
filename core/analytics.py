import pandas as pd

def annual_summary(series):
    x=series.resample("YE").mean()
    return pd.DataFrame({"Mean anomaly (°C)":x.values},index=x.index.year).rename_axis("Year")

def detect_records(series):
    x=series.resample("YE").mean().dropna()
    if x.empty: return pd.DataFrame()
    return pd.DataFrame([{"Record":"Warmest year in selected period","Year":x.idxmax().year,"Anomaly (°C)":x.max()},
                         {"Record":"Coolest year in selected period","Year":x.idxmin().year,"Anomaly (°C)":x.min()}])

def regional_series(grid,lat_min,lat_max,lon_min,lon_max):
    lat=grid["lat"]; lon=grid["lon"]
    lat_slice=slice(lat_min,lat_max) if lat.values[0]<lat.values[-1] else slice(lat_max,lat_min)
    sub=grid.sel(lat=lat_slice,lon=slice(lon_min,lon_max))
    return sub.mean(dim=["lat","lon"],skipna=True).to_pandas()
