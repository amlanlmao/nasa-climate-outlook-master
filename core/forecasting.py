import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

MODEL_NAMES=["Seasonal trend","Persistence","ARIMA","SARIMA","Exponential Smoothing","Random Forest","XGBoost"]

def clean_series(series):
    s=series.dropna().astype(float).sort_index()
    return s.asfreq("MS").interpolate(limit=2).dropna()

def seasonal_trend(s,h):
    recent=s.iloc[-min(180,len(s)):]; means=recent.groupby(recent.index.month).mean().reindex(range(1,13)).interpolate().bfill().ffill()
    n=min(120,len(recent)); slope=np.polyfit(np.arange(n),recent.iloc[-n:].values,1)[0]
    idx=pd.date_range(s.index[-1]+pd.offsets.MonthBegin(1),periods=h,freq="MS")
    return pd.Series([means.loc[d.month]+slope*(i+1) for i,d in enumerate(idx)],index=idx)

def rf_fit(s,h,kind="rf"):
    from sklearn.ensemble import RandomForestRegressor
    lag=12; vals=s.values
    X=np.array([vals[i-lag:i] for i in range(lag,len(vals))]); y=vals[lag:]
    if kind=="rf":
        model=RandomForestRegressor(n_estimators=220,min_samples_leaf=3,max_features=.8,random_state=42,n_jobs=-1)
    else:
        from xgboost import XGBRegressor
        model=XGBRegressor(n_estimators=220,max_depth=3,learning_rate=.03,subsample=.85,colsample_bytree=.85,objective="reg:squarederror",random_state=42,n_jobs=2)
    model.fit(X,y); history=list(vals[-lag:]); out=[]
    for _ in range(h):
        p=float(model.predict(np.array(history[-lag:]).reshape(1,-1))[0]); out.append(p); history.append(p)
    idx=pd.date_range(s.index[-1]+pd.offsets.MonthBegin(1),periods=h,freq="MS")
    return pd.Series(out,index=idx),model,(X,y)

def fit_predict(s,name,h):
    if name=="Seasonal trend": return seasonal_trend(s,h),None,None,"Seasonal means plus recent linear trend."
    if name=="Persistence":
        idx=pd.date_range(s.index[-1]+pd.offsets.MonthBegin(1),periods=h,freq="MS")
        return pd.Series(float(s.iloc[-1]),index=idx),None,None,"Repeats the latest observation."
    if name=="ARIMA":
        fit=ARIMA(s,order=(2,1,2)).fit(); pred=fit.get_forecast(h)
        return pred.predicted_mean,pred,None,"ARIMA(2,1,2); model-native prediction interval."
    if name=="SARIMA":
        fit=ARIMA(s,order=(1,1,1),seasonal_order=(1,0,1,12)).fit(); pred=fit.get_forecast(h)
        return pred.predicted_mean,pred,None,"Seasonal ARIMA(1,1,1) × (1,0,1,12)."
    if name=="Exponential Smoothing":
        fit=ExponentialSmoothing(s,trend="add",damped_trend=True,seasonal="add",seasonal_periods=12,initialization_method="estimated").fit()
        return fit.forecast(h),fit,None,"Damped additive trend and additive seasonality."
    if name in ("Random Forest","XGBoost"):
        fc,model,training=rf_fit(s,h,"rf" if name=="Random Forest" else "xgb")
        return fc,model,training,"Recursive 12-lag machine-learning model."
    raise ValueError(name)

def bootstrap_band(fc,residuals,alpha=.05,n=1000):
    r=np.asarray(residuals); r=r[np.isfinite(r)]
    if len(r)<8: r=np.array([-.1,.1])
    rng=np.random.default_rng(7); draws=rng.choice(r,(n,len(fc)),replace=True)
    paths=np.asarray(fc)[None,:]+np.cumsum(draws,axis=1)
    return pd.Series(np.quantile(paths,alpha/2,axis=0),index=fc.index),pd.Series(np.quantile(paths,1-alpha/2,axis=0),index=fc.index)

def forecast_model(series,name,horizon=12):
    s=clean_series(series)
    if len(s)<60: raise ValueError("At least 60 monthly observations are recommended.")
    fc,fit,training,note=fit_predict(s,name,horizon)
    if name in ("ARIMA","SARIMA"):
        ci=fit.conf_int(); lo=pd.Series(ci.iloc[:,0].values,index=fc.index); hi=pd.Series(ci.iloc[:,1].values,index=fc.index)
    elif name=="Exponential Smoothing":
        lo,hi=bootstrap_band(fc,fit.resid)
    elif name in ("Random Forest","XGBoost"):
        X,y=training; residual=y-fit.predict(X); lo,hi=bootstrap_band(fc,residual)
    else:
        lo,hi=bootstrap_band(fc,s.diff(12).dropna().values)
    return {"forecast":pd.Series(np.asarray(fc),index=fc.index),"lower":lo,"upper":hi,
            "note":note+" Intervals are approximate and are not physical climate projection ranges."}

def evaluate_models(series,models=MODEL_NAMES,folds=4,horizon=12):
    s=clean_series(series); origins=np.linspace(len(s)-folds*horizon,len(s)-horizon,num=folds,dtype=int); rows=[]
    for name in models:
        errors=[]; naive=[]; start=time.perf_counter(); failure=""
        for origin in origins:
            train=s.iloc[:origin]; actual=s.iloc[origin:origin+horizon]
            if len(actual)!=horizon: continue
            try:
                pred,_,_,_=fit_predict(train,name,horizon)
                errors.extend(np.abs(actual.values-np.asarray(pred)))
                naive.extend(np.abs(actual.values-np.repeat(train.iloc[-1],horizon)))
            except Exception as e: failure=str(e); break
        elapsed=time.perf_counter()-start
        if errors and not failure:
            mae=float(np.mean(errors)); rmse=float(np.sqrt(np.mean(np.square(errors)))); denom=float(np.mean(naive)); mase=mae/denom if denom else np.nan
        else: mae=rmse=mase=np.nan
        rows.append({"Model":name,"MAE (°C)":mae,"RMSE (°C)":rmse,"MASE":mase,"Seconds":elapsed,"Status":failure or ("OK" if errors else "Insufficient data")})
    return pd.DataFrame(rows).sort_values("MAE (°C)",na_position="last").reset_index(drop=True)
