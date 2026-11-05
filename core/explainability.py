import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def explain_random_forest(series):
    try: import shap
    except ImportError as e: raise RuntimeError("Install SHAP with: python -m pip install shap") from e
    s=series.dropna().astype(float); lag=12
    X=np.array([s.values[i-lag:i] for i in range(lag,len(s))]); y=s.values[lag:]
    names=[f"lag_{i}m" for i in range(lag,0,-1)]
    model=RandomForestRegressor(n_estimators=220,min_samples_leaf=3,max_features=.8,random_state=42,n_jobs=-1).fit(X,y)
    vals=shap.TreeExplainer(model).shap_values(X[-min(200,len(X)):])
    return pd.Series(np.abs(vals).mean(axis=0),index=names).sort_values(ascending=False)
