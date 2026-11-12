"""Time-based validation and calibrated risk estimators."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error
from .data import FEATURES

def ks(y,p):
    order=np.argsort(p)
    yy=np.asarray(y)[order]
    if yy.sum()==0 or yy.sum()==len(yy): return float('nan')
    return float(np.max(np.abs(np.cumsum(yy)/yy.sum()-np.cumsum(1-yy)/(len(yy)-yy.sum()))))

def calibration(y,p, bins=10):
    d=pd.DataFrame({'y':y,'p':p})
    d['bin']=pd.qcut(d['p'].rank(method='first'),q=min(bins,len(d)),labels=False)
    return d.groupby('bin').agg(n=('y','size'),actual=('y','mean'),predicted=('p','mean')).reset_index().to_dict('records')

def fit_models(df, output='outputs'):
    path=Path(output);path.mkdir(parents=True,exist_ok=True)
    # Application months 0..15 train; 16..19 calibration; 20..23 untouched OOT test.
    train=df[df.application_month<16]; cal=df[(df.application_month>=16)&(df.application_month<20)]; test=df[df.application_month>=20]
    models={}
    for name,target in [('pd','default_12m'),('takeup','accepted_offer')]:
        m=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=1500,C=0.5))
        m.fit(train[FEATURES],train[target]); models[name]=m
    for name,target in [('lgd','lgd_observed'),('ead_util','utilization_observed')]:
        m=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=100))
        # LGD model is fit only on defaulted accounts, simulating recovery observation.
        subset=train[train.default_12m==1] if name=='lgd' else train
        m.fit(subset[FEATURES],subset[target]);models[name]=m
    # Platt calibration using a separate temporal slice.
    raw=models['pd'].predict_proba(cal[FEATURES])[:,1]
    calibrator=LogisticRegression().fit(np.log(np.clip(raw,1e-6,1-1e-6)/(1-np.clip(raw,1e-6,1-1e-6))).reshape(-1,1),cal.default_12m)
    models['pd_calibrator']=calibrator
    def predict_pd(x):
        p=np.clip(models['pd'].predict_proba(x[FEATURES])[:,1],1e-6,1-1e-6)
        return calibrator.predict_proba(np.log(p/(1-p)).reshape(-1,1))[:,1]
    pred=predict_pd(test)
    metrics={'sample_sizes':{'train':len(train),'calibration':len(cal),'out_of_time_test':len(test)},
        'pd':{'auc':float(roc_auc_score(test.default_12m,pred)),'ks':ks(test.default_12m,pred),
              'brier':float(brier_score_loss(test.default_12m,pred)), 'calibration':calibration(test.default_12m,pred)},
        'lgd_mae_defaults':float(mean_absolute_error(test.loc[test.default_12m==1,'lgd_observed'],np.clip(models['lgd'].predict(test.loc[test.default_12m==1,FEATURES]),.05,1))),
        'ead_util_mae':float(mean_absolute_error(test.utilization_observed,np.clip(models['ead_util'].predict(test[FEATURES]),.05,1))),
        'takeup_auc':float(roc_auc_score(test.accepted_offer,models['takeup'].predict_proba(test[FEATURES])[:,1]))}
    dump(models,path/'models.joblib');(path/'validation.json').write_text(json.dumps(metrics,indent=2))
    return models,metrics

def load_models(output='outputs'):
    return load(Path(output)/'models.joblib')

def predict(models,record):
    x=pd.DataFrame([record]).reindex(columns=FEATURES)
    raw=float(models['pd'].predict_proba(x)[0,1]); raw=np.clip(raw,1e-6,1-1e-6)
    p=float(models['pd_calibrator'].predict_proba([[np.log(raw/(1-raw))]])[0,1])
    return {'pd':p,'lgd':float(np.clip(models['lgd'].predict(x)[0],.1,1)),
        'ead_util':float(np.clip(models['ead_util'].predict(x)[0],.05,1)),
        'takeup':float(np.clip(models['takeup'].predict_proba(x)[0,1],.05,.95))}
