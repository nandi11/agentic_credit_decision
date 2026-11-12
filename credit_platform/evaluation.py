"""Descriptive fairness and champion/challenger diagnostics, not legal compliance certification."""
import pandas as pd
import numpy as np
from .agents import UnderwritingOrchestrator

def portfolio_evaluation(df,models,limit=400):
    # Audit group is NOT supplied to underwriting orchestrator.
    sample=df[df.application_month>=20].head(limit).copy()
    engine=UnderwritingOrchestrator(models)
    rows=[]
    for _,r in sample.iterrows():
        x=r.to_dict();group=x.pop('audit_group'); result=engine.run(x)
        champion=(x['bureau_score']>=680 and x['dti']<=.45 and x['delinquencies']<=1)
        offer=result.get('best_offer')
        rows.append({'application_id':x['application_id'],'audit_group':group,'challenger_approve':int(result['decision']=='APPROVE'),
                     'champion_approve':int(champion),'predicted_pd':result.get('risk',{}).get('pd',np.nan),
                     'expected_value':offer['expected_value'] if offer else 0,'observed_default':x['default_12m']})
    out=pd.DataFrame(rows)
    groups=out.groupby('audit_group').agg(n=('challenger_approve','size'),challenger_approval_rate=('challenger_approve','mean'),champion_approval_rate=('champion_approve','mean'),mean_predicted_pd=('predicted_pd','mean'),observed_default_rate=('observed_default','mean')).reset_index()
    rates=groups.challenger_approval_rate.to_numpy();ratio=float(min(rates)/max(rates)) if max(rates)>0 else None
    summary={'n':len(out),'champion_approval_rate':float(out.champion_approve.mean()),'challenger_approval_rate':float(out.challenger_approve.mean()),
        'challenger_sum_expected_value':float(out.expected_value.sum()),'approval_rate_ratio_min_over_max':ratio,
        'caveats':['Synthetic audit_group is not an actual protected class or proxy','Approval ratio is a screening metric, not a legal test','Observed defaults reflect historical synthetic outcomes, not causal impact of offers','Champion and challenger economics are not directly comparable without matched offer assumptions']}
    return out,groups,summary
