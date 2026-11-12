"""Entirely fictional applicant and outcome simulator; not a population model."""
import numpy as np
import pandas as pd

FEATURES = ['bureau_score','utilization','delinquencies','inquiries','annual_income','dti','cashflow_buffer','nsf_count','employment_years','existing_relationship']
AUDIT = 'audit_group'  # synthetic, held separately from decision features

def generate(n=10000, seed=42):
    r = np.random.default_rng(seed)
    age = r.integers(21,76,n)
    group = r.choice(['Group A','Group B'],n,p=[.57,.43])
    income = np.clip(r.lognormal(np.log(68000),.55,n),18000,260000).round()
    score = np.clip(r.normal(695,78,n) + (group=='Group B')*(-12),450,850).round()
    util = np.clip(r.beta(2.1,3.8,n)+ (group=='Group B')*.035,0,1)
    dti = np.clip(r.beta(2.4,4.5,n)*1.05,0,.95)
    delin = np.clip(r.poisson(np.clip((690-score)/110+.45,.1,2.5)),0,5)
    inquiries = r.poisson(1.3,n)
    cash = np.clip(r.normal(.18,.19,n)+(score-690)/1800,-.55,.9)
    nsf = r.poisson(np.clip(.65-cash, .08,2.3))
    employment = np.clip(r.exponential(6,n),0,35)
    relation = r.binomial(1,.35,n)
    df = pd.DataFrame({'application_id':[f'APP{i:06d}' for i in range(n)], 'application_month':np.arange(n)%24,
        'age':age,'bureau_score':score,'utilization':util.round(4),'delinquencies':delin,'inquiries':inquiries,
        'annual_income':income,'dti':dti.round(4),'cashflow_buffer':cash.round(4),'nsf_count':nsf,
        'employment_years':employment.round(2),'existing_relationship':relation,'audit_group':group})
    # Outcome simulator: only features allowed in model, plus independent noise.
    z = -2.65 + (685-score)/85 + 1.45*util + 1.25*dti + .28*delin + .12*inquiries - 1.2*cash + .16*nsf - .000004*(income-65000)
    pd_true = np.clip(1/(1+np.exp(-z)),.004,.85)
    df['default_12m'] = r.binomial(1,pd_true)
    df['lgd_observed'] = np.clip(.65+.14*util+.10*dti-.12*cash+r.normal(0,.12,n),.15,1.0)
    df['utilization_observed'] = np.clip(.25+.43*util+.18*dti-.10*cash+r.normal(0,.10,n),.05,1)
    df['accepted_offer'] = r.binomial(1,np.clip(.65-.10*(score-700)/150+.08*cash,.2,.9))
    return df
