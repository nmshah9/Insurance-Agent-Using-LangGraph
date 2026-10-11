# Insurance Claim Processing Agent (LangGraph + Streamlit)

## Workflow

```
START -> intake -> +-- document_verification --+
                   +-- eligibility_check ------+--> decision_engine -> claim_summary
                   +-- fraud_detection --------+                          |
                         (parallel)                    (conditional edge) |
                          auto_approve <-- approve ----+---- reject --> auto_reject
                                                       +-- escalate --> human_approval (interrupt)
```

| Requirement | Where |
|---|---|
| LangGraph | `claims_agent.py` -> `build_graph()` |
| 5 agents | Document Verification, Eligibility Check, Fraud Detection, Claim Summary, Human Approval |
| Parallel execution | three edges fan out from `intake`; one list-edge joins them at `decision_engine` |
| Conditional edges | `add_conditional_edges("claim_summary", route_decision, ...)` |
| Human-in-the-loop | `interrupt()` in `human_approval_agent`; resumed with `Command(resume=...)` |
| LLM | GPT-4o-mini in Fraud Detection (second opinion) and Claim Summary; rule-based fallback if no key |

## Decision rules

- **Reject:** missing required documents, or any hard eligibility failure (expired / inactive policy, holder mismatch, incident outside policy period, type not covered).
- **Human review:** claim >= Rs 5,00,000, fraud score >= 40, or claim exceeds remaining coverage.
- **Auto-approve:** everything else.

## Setup (Windows CMD)

**Checkpoint 1 - environment**
```
cd insurance_claim_agent
python -m venv claimenv
claimenv\Scripts\activate
pip install -r requirements.txt
```

**Checkpoint 2 - verify the graph (no API key needed)**
```
python test_scenarios.py
```
Expect 5 x `PASS` and `ALL SCENARIOS PASSED`.

**Checkpoint 3 - run the app**
```
streamlit run app.py
```
Click each scenario in the sidebar, then **Process claim**. Scenarios 3 and 5 pause for human approval.
To use the LLM, paste your OpenAI key in the sidebar (or set `OPENAI_API_KEY`).

## Deploy to Streamlit Community Cloud

1. Push this folder to GitHub (`app.py` and `requirements.txt` at the repo root).
2. share.streamlit.io -> New app -> select repo -> main file `app.py`.
3. App settings -> Secrets:
   ```
   OPENAI_API_KEY = "sk-..."
   ```
