import numpy as np
import pandas as pd
from core.forecasting import forecast_model

def test_baseline_horizon():
    idx=pd.date_range("1980-01-01",periods=600,freq="MS")
    s=pd.Series(.2*np.sin(2*np.pi*idx.month/12)+np.arange(600)*.0005,index=idx)
    for name in ["Seasonal trend","Persistence"]:
        r=forecast_model(s,name,6)
        assert len(r["forecast"])==6
        assert len(r["lower"])==6
        assert len(r["upper"])==6
