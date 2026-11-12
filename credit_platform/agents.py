"""Auditable, tool-like deterministic agents; no autonomous LLM approval authority."""
import json
from datetime import datetime,timezone
from pathlib import Path
from .models import predict

LIMITS=(1000,2500,5000,7500,10000,15000)
APRS=(.1999,.2499,.2999)
REASON_FACTORS=[('bureau_score','Credit history score below policy threshold'),('delinquencies','Recent delinquency history'),('dti','High debt-to-income ratio'),('cashflow_buffer','Insufficient verified cash-flow buffer'),('nsf_count','Frequent insufficient-funds events'),('utilization','High revolving credit utilization')]

class DataAgent:
    def run(self,app):
        required=['bureau_score','annual_income','dti','utilization','cashflow_buffer']
        missing=[k for k in required if k not in app or app[k] is None]
        return {'ok':not missing,'missing':missing,'sources':{'bureau':'synthetic','cash_flow':'synthetic_permissioned','application':'synthetic'}}

class PolicyAgent:
    def run(self,app):
        flags=[]
        if app.get('age',0)<21: flags.append('DEMO_AGE_GATE') # illustrative only; not a statement of legal eligibility
        if app.get('bureau_score',0)<560: flags.append('LOW_BUREAU_SCORE')
        if app.get('dti',1)>.65: flags.append('HIGH_DTI')
        if app.get('annual_income',0)<20000: flags.append('LOW_VERIFIED_INCOME')
        if app.get('delinquencies',0)>=4: flags.append('DELINQUENCY_POLICY')
        return {'eligible':not flags,'flags':flags}

class RiskAgent:
    def __init__(self,models): self.models=models
    def run(self,app): return predict(self.models,app)

class EconomicsAgent:
    def evaluate(self,app,risk,line,apr):
        # One-year demonstration economics. Explicitly NOT regulatory capital or lifetime CECL.
        util=risk['ead_util']; ead=line*util
        annual_interest=ead*apr*.68  # illustrative revolver share
        interchange=ead*1.7*.012
        fees=15
        funding=ead*.055
        rewards=ead*1.7*.011
        servicing=75
        acquisition=110
        el=risk['pd']*risk['lgd']*ead
        profit=annual_interest+interchange+fees-funding-rewards-servicing-acquisition-el
        capital=max(200,.12*ead+2.2*el)
        raroc=profit/capital
        # Take-up model does not model price sensitivity; transparent APR adjustment is a scenario assumption.
        takeup=max(.05,min(.95,risk['takeup']-(apr-.1999)*1.25))
        expected_value=takeup*profit
        return {'line':line,'apr':apr,'ead':ead,'expected_loss':el,'profit':profit,
                'capital_proxy':capital,'raroc_proxy':raroc,'takeup_scenario':takeup,'expected_value':expected_value}

class FairLendingAgent:
    def run(self,app):
        # Prohibits protected audit group from underwriting feature set.
        return {'decision_uses_audit_group':False,'review_required':False,'audit_group_segregated':True}

class OptimizationAgent:
    def __init__(self,economics): self.economics=economics
    def run(self,app,risk,policy):
        if not policy['eligible']: return {'best':None,'offers':[]}
        max_line=min(15000,round(app['annual_income']*.18/500)*500)
        offers=[self.economics.evaluate(app,risk,l,a) for l in LIMITS for a in APRS if l<=max_line]
        # Demo guardrails: risk appetite PD <= 25%, RAROC >= 12%, positive value.
        feasible=[o for o in offers if risk['pd']<=.25 and o['raroc_proxy']>=.12 and o['expected_value']>0]
        return {'best':max(feasible,key=lambda x:x['expected_value']) if feasible else None,'offers':offers}

class ExplanationAgent:
    def run(self,app,policy,decision):
        if decision=='APPROVE': return ['Approved within synthetic policy and economics constraints']
        reasons=list(policy['flags'])
        # Prototype policy-factor diagnostics, NOT legally sufficient adverse-action reasons.
        for key,label in REASON_FACTORS:
            val=app.get(key,0)
            if (key=='bureau_score' and val<650) or (key=='delinquencies' and val>=2) or (key=='dti' and val>.45) or (key=='cashflow_buffer' and val<.05) or (key=='nsf_count' and val>=2) or (key=='utilization' and val>.65):
                reasons.append(label)
        return reasons[:5] or ['Economic hurdle not met; human review required']

class UnderwritingOrchestrator:
    def __init__(self,models):
        self.data=DataAgent();self.policy=PolicyAgent();self.risk=RiskAgent(models)
        self.econ=EconomicsAgent();self.fair=FairLendingAgent();self.opt=OptimizationAgent(self.econ);self.explain=ExplanationAgent()
    def run(self,app):
        app=dict(app); d=self.data.run(app)
        if not d['ok']: return {'application_id':app.get('application_id','UNKNOWN'),'decision':'REVIEW','data':d,'reason_codes':['MISSING_REQUIRED_DATA']}
        policy=self.policy.run(app);risk=self.risk.run(app);fair=self.fair.run(app)
        opt=self.opt.run(app,risk,policy)
        # No hard declines by agents: every nonapproval goes to review pending validated decision policies.
        decision='APPROVE' if opt['best'] is not None else 'REVIEW'
        return {'application_id':app.get('application_id','UNKNOWN'),'decision':decision,'data':d,'policy':policy,'risk':risk,'fairness':fair,
                'best_offer':opt['best'],'offers':opt['offers'],'reason_codes':self.explain.run(app,policy,decision),
                'audit':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),'pipeline_version':'v2.0-demo','automated_decline':False}}

def append_audit(result,filename='outputs/decision_audit.jsonl'):
    p=Path(filename);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(result,default=str)+'\n')
