from io import StringIO
import os, tempfile
import pandas as pd
import requests
import xarray as xr

NASA_URL="https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
NOAA_URL="https://psl.noaa.gov/data/timeseries/month/data/nasa.land.csv"
GRID_URL="https://data.giss.nasa.gov/pub/gistemp/gistemp250_GHCNv4_ERSSTv5.nc.gz"
HEADERS={"User-Agent":"NASAClimateOutlook/2.0 educational dashboard"}

def load_global_temperature():
    r=requests.get(NASA_URL,headers=HEADERS,timeout=25); r.raise_for_status()
    raw=pd.read_csv(StringIO(r.text),skiprows=1).rename(columns={"Year":"Year"})
    raw=raw.rename(columns={raw.columns[0]:"Year"}); raw["Year"]=pd.to_numeric(raw["Year"],errors="coerce")
    months=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    rows=[]
    for _,row in raw.dropna(subset=["Year"]).iterrows():
        for mnum,m in enumerate(months,1):
            val=pd.to_numeric(row.get(m),errors="coerce")
            if pd.notna(val): rows.append((pd.Timestamp(int(row["Year"]),mnum,1),float(val)))
    df=pd.DataFrame(rows,columns=["date","anomaly_c"]).set_index("date").sort_index()
    if df.empty: raise ValueError("NASA GISTEMP returned no observations.")
    df.attrs.update(source="NASA GISTEMP v4 land-ocean",baseline="1951–1980")
    return df

def load_alternative_temperature():
    r=requests.get(NOAA_URL,headers=HEADERS,timeout=25); r.raise_for_status()
    df=pd.read_csv(StringIO(r.text),skiprows=1,names=["date","anomaly_c"])
    df["date"]=pd.to_datetime(df["date"],errors="coerce"); df["anomaly_c"]=pd.to_numeric(df["anomaly_c"],errors="coerce")
    df=df[df["anomaly_c"].between(-20,20)].dropna().set_index("date").sort_index()
    if df.empty: raise ValueError("NOAA mirror returned no observations.")
    df.attrs.update(source="NOAA PSL mirror of NASA land series",baseline="Provider-specific; verify before comparison")
    return df

def load_gridded_temperature():
    r=requests.get(GRID_URL,headers=HEADERS,timeout=180); r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".nc.gz",delete=False) as f:
        f.write(r.content); path=f.name
    try:
        ds=xr.open_dataset(path)
        var=next((v for v in ds.data_vars if ds[v].ndim>=3),None)
        if var is None: raise ValueError(f"No 3D variable in NetCDF. Variables: {list(ds.data_vars)}")
        da=ds[var]
        ren={}
        for dim in da.dims:
            d=dim.lower()
            if d in ("latitude","latitude1","y"): ren[dim]="lat"
            elif d in ("longitude","longitude1","x"): ren[dim]="lon"
            elif d in ("time","month","date"): ren[dim]="time"
        da=da.rename(ren)
        if not {"lat","lon","time"}.issubset(da.dims): raise ValueError(f"Unexpected NetCDF dimensions: {da.dims}")
        return da
    finally:
        try: os.unlink(path)
        except OSError: pass
