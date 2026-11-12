"""python run_pipeline.py --n 20000"""
import argparse,json
from pathlib import Path
from credit_platform.data import generate
from credit_platform.models import fit_models
from credit_platform.evaluation import portfolio_evaluation
from credit_platform.agents import UnderwritingOrchestrator,append_audit

def main():
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=20000);p.add_argument('--seed',type=int,default=42);args=p.parse_args()
    out=Path('outputs');out.mkdir(exist_ok=True)
    df=generate(args.n,args.seed);df.to_csv(out/'synthetic_applications.csv',index=False)
    models,metrics=fit_models(df,out)
    decisions,groups,summary=portfolio_evaluation(df,models)
    decisions.to_csv(out/'champion_challenger.csv',index=False)
    groups.to_csv(out/'fairness_audit.csv',index=False)
    (out/'portfolio_summary.json').write_text(json.dumps(summary,indent=2))
    sample=df.iloc[-1].drop('audit_group').to_dict();result=UnderwritingOrchestrator(models).run(sample)
    append_audit(result,out/'decision_audit.jsonl')
    print(json.dumps({'validation':{k:v for k,v in metrics.items() if k!='pd'},'pd':{k:v for k,v in metrics['pd'].items() if k!='calibration'},'portfolio':summary,'sample_decision':result['decision']},indent=2))
if __name__=='__main__':main()
