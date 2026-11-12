# Agentic Credit Acquisition Platform — v2 Research Simulator

An executable, fully synthetic, **non-production** demonstration of credit-card application decision orchestration, calibrated 12-month PD modeling, LGD/EAD utilization and take-up models, offer economics, constrained optimization, policy controls, and fairness screening.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python run_pipeline.py --n 20000
streamlit run app.py
pytest -q
```

Run from the unzipped project root. The pipeline generates all outputs locally in `outputs/` and requires no network services after dependencies are installed.

## Architecture

`DataAgent` verifies synthetic data completeness and labels sources. `RiskAgent` calls four fitted statistical models. `PolicyAgent` enforces demonstration constraints. `EconomicsAgent` computes one-year illustrative revenue, credit loss and capital proxy. `OptimizationAgent` evaluates 18 APR/limit scenarios and maximizes acceptance-weighted profit under hard gates. `FairLendingAgent` ensures the fictional audit group is excluded from underwriting. `ExplanationAgent` emits diagnostic factors. `UnderwritingOrchestrator` produces a structured trace and sends all nonapprovals to **REVIEW**, not automated decline.

These are deterministic software agents/services, **not LLM agents**. This is intentional: no generative model is permitted to independently approve/decline or invent credit reasons. For an LLM extension, implement a read-only, schema-validated assistant that can query approved services and summarize their traces, never override policy or generate unverified adverse-action notices.

## Outputs

- `synthetic_applications.csv`: 20,000 fictional records with 24 simulated application months and observed outcomes.
- `models.joblib`: fitted models, including Platt PD calibration on a held-out temporal window.
- `validation.json`: out-of-time AUC, KS, Brier, calibration, LGD/EAD MAE and take-up AUC.
- `champion_challenger.csv`: paired policy decisions on untouched OOT cases.
- `fairness_audit.csv`: group approval rates and outcome diagnostics.
- `portfolio_summary.json`: aggregate strategy comparison.
- `decision_audit.jsonl`: append-only-style local demo trace (not tamper-evident).

## Key methodological limitations

1. **Synthetic data** are generated from assumed equations; model performance does not imply real-world accuracy. Synthetic group disparities are intentionally included to exercise audit logic.
2. The PD is a **12-month default** model, not lifetime CECL ECL. LGD observations are simulated and LGD training uses only default cases. EAD is a utilization proxy; no CCF, dynamic balances or competing risks.
3. The offer take-up model is trained on a synthetic binary outcome **without observed APR/line variation**. APR sensitivity is a disclosed scenario assumption, **not causally identified**. Pricing and credit-line optimization therefore cannot be used for real offers.
4. The capital denominator is a proxy, not regulatory or validated economic capital. The objective is expected value, not a mathematically optimized portfolio RAROC.
5. The fairness screen is a diagnostic only; it is not an ECOA/Reg B legal conclusion. No group data enter underwriting. Group labels are fictional, not actual protected-class information.
6. The reason strings are **not compliant adverse-action notices** and are not guaranteed to reflect the exact marginal decision factors. Production requires verified decision-causal reason codes, adverse-action workflows, legal review and human escalation.
7. There is no bureau access, permissible-purpose check, consumer consent system, KYC/CIP, fraud screening, identity verification, GLBA security, production monitoring, model registry, robust logging, authentication, audit retention or independent validation.
8. The champion is a simple score-based benchmark, not a fully matched economics strategy; compare approval rates, not realized profitability or causal uplift.
9. Data are sampled across 24 simulated months with static data-generating assumptions; OOT tests exercise split mechanics but do not establish real temporal robustness.

## Production roadmap

Replace synthetic inputs with permissioned and contractually authorized sources; establish application and outcome timestamp lineage; train PD, LGD, EAD, behavior, acceptance and fraud models on representative observed data; validate against selection bias, reject inference, drift and calibration; build verified adverse-action factor mapping; implement ability-to-pay and pricing legal controls; conduct fair-lending testing; obtain independent model validation and governance approval; run shadow mode before controlled production rollout.

**No consumer decision should be made using this project.**
