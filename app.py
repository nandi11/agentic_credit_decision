"""Run: streamlit run app.py"""
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
from credit_platform.models import load_models
from credit_platform.agents import UnderwritingOrchestrator,append_audit

st.set_page_config(page_title='Credit Acquisition AI Lab',layout='wide')
st.title('Credit Acquisition AI Lab')
st.caption('Research simulator only — no real applications, credit decisions, adverse-action notices or compliance certification.')
if not Path('outputs/models.joblib').exists():
    st.error('First run: python run_pipeline.py');st.stop()
@st.cache_resource
def get_models():return load_models('outputs')
models=get_models();engine=UnderwritingOrchestrator(models)
t1,t2,t3=st.tabs(['Application decision','Portfolio & fairness','Validation'])
with t1:
    with st.form('application'):
        a,b,c=st.columns(3)
        with a:
            score=st.slider('Bureau score',450,850,705);income=st.number_input('Annual income ($)',20000,260000,75000,step=5000);dti=st.slider('Debt-to-income',0.,.95,.28,.01);age=st.number_input('Age',21,85,35)
        with b:
            util=st.slider('Revolving utilization',0.,1.,.34,.01);delin=st.number_input('Delinquencies',0,5,0);inq=st.number_input('Inquiries',0,12,1);cash=st.slider('Cash-flow buffer',-.55,.9,.22,.01)
        with c:
            nsf=st.number_input('NSF events',0,10,0);employment=st.number_input('Employment years',0.,40.,6.);relationship=st.checkbox('Existing relationship',value=True)
        submitted=st.form_submit_button('Run decision pipeline',type='primary')
    if submitted:
        app={'application_id':'DEMO-UI','age':age,'bureau_score':score,'annual_income':income,'dti':dti,'utilization':util,'delinquencies':delin,'inquiries':inq,'cashflow_buffer':cash,'nsf_count':nsf,'employment_years':employment,'existing_relationship':int(relationship)}
        result=engine.run(app);append_audit(result)
        st.subheader('Decision: '+result['decision']);r=result['risk'];st.metric('Predicted 12-month PD',f"{r['pd']:.2%}")
        st.write('Model outputs',r);st.write('Policy',result['policy']);st.write('Diagnostic factors (not adverse-action reasons)',result['reason_codes'])
        if result['best_offer']:
            st.success('Highest expected-value eligible offer');st.json(result['best_offer'])
        if result['offers']:
            o=pd.DataFrame(result['offers']);o['APR']=o.apr.map(lambda x:f'{x:.2%}')
            st.dataframe(o[['line','APR','expected_loss','profit','capital_proxy','raroc_proxy','takeup_scenario','expected_value']],hide_index=True)
            st.plotly_chart(px.scatter(o,x='line',y='expected_value',color='APR',title='Scenario expected value by credit line'),use_container_width=True)
        st.download_button('Download decision trace',data=__import__('json').dumps(result,indent=2,default=str),file_name='decision_trace.json')
with t2:
    p=Path('outputs/champion_challenger.csv');f=Path('outputs/fairness_audit.csv')
    if p.exists():
        df=pd.read_csv(p);groups=pd.read_csv(f)
        c1,c2=st.columns(2)
        c1.metric('Traditional champion approval',f'{df.champion_approve.mean():.1%}')
        c2.metric('Optimized challenger approval',f'{df.challenger_approve.mean():.1%}')
        st.plotly_chart(px.bar(groups,x='audit_group',y=['champion_approval_rate','challenger_approval_rate'],barmode='group',title='Approval rate by fictional audit group'),use_container_width=True)
        st.dataframe(groups,hide_index=True)
        st.warning('Fairness metrics are descriptive screens, not a disparate-impact determination. Audit groups are synthetic and never used as decision features.')
with t3:
    import json
    m=json.loads(Path('outputs/validation.json').read_text())
    a,b,c=st.columns(3)
    a.metric('OOT PD AUC',f"{m['pd']['auc']:.3f}");b.metric('OOT KS',f"{m['pd']['ks']:.3f}");c.metric('OOT Brier',f"{m['pd']['brier']:.4f}")
    st.plotly_chart(px.line(pd.DataFrame(m['pd']['calibration']),x='predicted',y='actual',markers=True,title='Out-of-time PD calibration'),use_container_width=True)
    st.json({k:v for k,v in m.items() if k!='pd'})
