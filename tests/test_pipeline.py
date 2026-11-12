from credit_platform.data import generate,FEATURES
from credit_platform.models import fit_models
from credit_platform.agents import UnderwritingOrchestrator

def test_synthetic_schema():
    df=generate(120,3)
    assert all(c in df.columns for c in FEATURES)
    assert df.default_12m.isin([0,1]).all()

def test_no_audit_feature():
    assert 'audit_group' not in FEATURES

def test_pipeline(tmp_path):
    df=generate(1200,9);m,metrics=fit_models(df,tmp_path)
    app=df.iloc[0].drop('audit_group').to_dict();r=UnderwritingOrchestrator(m).run(app)
    assert r['decision'] in ['APPROVE','REVIEW']
    assert 0<r['risk']['pd']<1
    assert all(o['line']>0 and o['capital_proxy']>0 for o in r['offers'])
    assert r['fairness']['decision_uses_audit_group'] is False
    assert 'auc' in metrics['pd']
