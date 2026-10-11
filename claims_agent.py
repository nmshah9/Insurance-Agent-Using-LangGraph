"""
Insurance Claim Processing Agent - LangGraph workflow.

Flow
----
START -> intake
      -> [document_verification | eligibility_check | fraud_detection]   (PARALLEL fan-out)
      -> decision_engine                                                (fan-in / join)
      -> claim_summary
      -> (conditional edge) auto_approve | auto_reject | human_approval  (HITL via interrupt)
      -> END

The LLM (GPT-4o-mini by default) is used by the Fraud Detection and Claim Summary
agents. If no OPENAI_API_KEY is set, both agents fall back to deterministic
logic so the app still runs end-to-end.
"""
import json
import operator
import os
import re
from datetime import date, datetime, timedelta
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Business configuration
# --------------------------------------------------------------------------- #
HIGH_VALUE_THRESHOLD = 500_000   # INR - valid claims at/above this need a human
FRAUD_ESCALATE_AT = 40           # fraud score at/above this -> human review
FRAUD_HIGH_AT = 60               # label only: "High" risk

REQUIRED_DOCS = {
    "Health": ["Claim Form", "Hospital Bill", "Discharge Summary", "ID Proof"],
    "Motor": ["Claim Form", "Police Report/FIR", "Repair Estimate", "Driving Licence", "Vehicle RC"],
    "Property": ["Claim Form", "Photos of Damage", "Repair Estimate", "Ownership Proof"],
}
ALL_DOC_TYPES = sorted({d for docs in REQUIRED_DOCS.values() for d in docs})

# Typical claim size per category (used by the fraud rules)
TYPICAL_AMOUNT = {"Health": 150_000, "Motor": 120_000, "Property": 250_000}

# Filename keywords -> document type (so uploaded files count as documents)
DOC_KEYWORDS = {
    "Claim Form": ["claim_form", "claimform", "claim-form", "claim form"],
    "Hospital Bill": ["hospital", "bill", "invoice"],
    "Discharge Summary": ["discharge"],
    "ID Proof": ["id_proof", "idproof", "aadhaar", "aadhar", "passport", "pan"],
    "Police Report/FIR": ["fir", "police"],
    "Repair Estimate": ["estimate", "estimates", "repair", "quotation"],
    "Driving Licence": ["licence", "license", "dl"],
    "Vehicle RC": ["rc", "registration"],
    "Photos of Damage": ["photo", "photos", "damage", "image", "images"],
    "Ownership Proof": ["ownership", "deed", "title"],
}

SUSPICIOUS_WORDS = [
    "urgent", "cash", "no witness", "total loss", "lost receipt",
    "settle quickly", "immediately", "cannot provide",
]


def _d(days_from_today: int) -> str:
    return (date.today() + timedelta(days=days_from_today)).isoformat()


# Mock policy database (dates are relative to today so the demo never goes stale)
POLICIES = {
    "POL-1001": {"holder": "Asha Mehta", "status": "Active", "start": _d(-200), "end": _d(165),
                 "sum_insured": 500_000, "claimed_ytd": 0, "prior_claims_12m": 0, "covers": ["Health"]},
    "POL-2002": {"holder": "Rohan Iyer", "status": "Active", "start": _d(-300), "end": _d(65),
                 "sum_insured": 800_000, "claimed_ytd": 0, "prior_claims_12m": 0, "covers": ["Motor"]},
    "POL-3003": {"holder": "Kiran Patel", "status": "Active", "start": _d(-455), "end": _d(-90),
                 "sum_insured": 600_000, "claimed_ytd": 0, "prior_claims_12m": 0, "covers": ["Health", "Property"]},
    "POL-4004": {"holder": "Meera Nair", "status": "Active", "start": _d(-120), "end": _d(245),
                 "sum_insured": 2_500_000, "claimed_ytd": 0, "prior_claims_12m": 0, "covers": ["Health"]},
    "POL-6006": {"holder": "Vikram Desai", "status": "Active", "start": _d(-180), "end": _d(185),
                 "sum_insured": 1_500_000, "claimed_ytd": 0, "prior_claims_12m": 0, "covers": ["Property"]},
    "POL-7007": {"holder": "Neha Kulkarni", "status": "Active", "start": _d(-150), "end": _d(215),
                 "sum_insured": 400_000, "claimed_ytd": 300_000, "prior_claims_12m": 1, "covers": ["Health"]},
    "POL-5005": {"holder": "Sanjay Rao", "status": "Active", "start": _d(-25), "end": _d(340),
                 "sum_insured": 800_000, "claimed_ytd": 0, "prior_claims_12m": 3, "covers": ["Motor"]},
}


def infer_docs_from_filenames(filenames: list[str]) -> list[str]:
    """Match whole words/phrases in the file name (so 'fire_damage' does not match 'fir')."""
    found = set()
    for name in filenames:
        stem = name.rsplit(".", 1)[0].lower()
        norm = " " + " ".join(re.split(r"[^a-z0-9]+", stem)).strip() + " "
        for doc_type, words in DOC_KEYWORDS.items():
            for w in words:
                kw = " " + " ".join(re.split(r"[^a-z0-9]+", w.lower())).strip() + " "
                if kw in norm:
                    found.add(doc_type)
    return sorted(found)


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class ClaimState(TypedDict, total=False):
    claim: dict
    doc_result: dict
    eligibility_result: dict
    fraud_result: dict
    decision: str                 # auto_approve | auto_reject | human_review
    decision_reasons: list
    summary: str
    final_decision: str           # APPROVED | REJECTED
    resolved_by: str
    human_review: dict
    audit_log: Annotated[list, operator.add]   # reducer: parallel nodes can all append


def _log(node: str, msg: str) -> str:
    return f"{datetime.now():%H:%M:%S.%f}"[:-3] + f" | {node} | {msg}"


# --------------------------------------------------------------------------- #
# LLM helper
# --------------------------------------------------------------------------- #
def get_llm():
    if not os.getenv("OPENAI_API_KEY") or os.getenv("CLAIMS_USE_LLM", "1") == "0":
        return None
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


class FraudOpinion(BaseModel):
    concern: bool
    rationale: str


# --------------------------------------------------------------------------- #
# Nodes / agents
# --------------------------------------------------------------------------- #
def intake(state: ClaimState):
    c = state["claim"]
    return {"audit_log": [_log("Intake", f"Received {c['claim_type']} claim on {c['policy_number']} "
                                         f"for Rs {c['amount']:,.0f}")]}


def document_verification_agent(state: ClaimState):
    claim = state["claim"]
    required = REQUIRED_DOCS.get(claim["claim_type"], [])
    provided = set(claim.get("documents", []))
    missing = [d for d in required if d not in provided]
    result = {
        "required": required,
        "provided": [d for d in required if d in provided],
        "missing": missing,
        "complete": not missing,
    }
    msg = "all required documents present" if not missing else f"missing: {', '.join(missing)}"
    return {"doc_result": result, "audit_log": [_log("Document Verification", msg)]}


def eligibility_check_agent(state: ClaimState):
    claim = state["claim"]
    policy = POLICIES.get(claim["policy_number"].strip().upper())
    checks = []

    def add(name, passed, detail, hard=True):
        checks.append({"check": name, "passed": passed, "detail": detail, "hard": hard})

    if not policy:
        add("Policy exists", False, f"No policy found for {claim['policy_number']}")
        result = {"eligible": False, "needs_review": False, "checks": checks, "remaining_coverage": 0}
        return {"eligibility_result": result,
                "audit_log": [_log("Eligibility Check", "policy not found")]}

    incident = date.fromisoformat(claim["incident_date"])
    start, end = date.fromisoformat(policy["start"]), date.fromisoformat(policy["end"])
    remaining = policy["sum_insured"] - policy["claimed_ytd"]

    add("Policy exists", True, f"{claim['policy_number']} held by {policy['holder']}")
    add("Policy holder matches", claim["claimant"].strip().lower() == policy["holder"].lower(),
        f"Claimant '{claim['claimant']}' vs holder '{policy['holder']}'")
    add("Policy status active", policy["status"] == "Active", f"Status: {policy['status']}")
    add("Policy not expired", end >= date.today(), f"Policy period {start} to {end}")
    add("Incident within policy period", start <= incident <= end,
        f"Incident {incident} vs period {start} to {end}")
    add("Incident date not in future", incident <= date.today(), f"Incident date {incident}")
    add("Claim type covered", claim["claim_type"] in policy["covers"],
        f"Covers: {', '.join(policy['covers'])}")
    add("Within remaining coverage", claim["amount"] <= remaining,
        f"Claim Rs {claim['amount']:,.0f} vs remaining Rs {remaining:,.0f}", hard=False)

    hard_fail = any(not c["passed"] and c["hard"] for c in checks)
    soft_fail = any(not c["passed"] and not c["hard"] for c in checks)
    result = {"eligible": not hard_fail, "needs_review": soft_fail and not hard_fail,
              "checks": checks, "remaining_coverage": remaining}
    msg = "eligible" if result["eligible"] else "NOT eligible: " + "; ".join(
        c["check"] for c in checks if not c["passed"] and c["hard"])
    return {"eligibility_result": result, "audit_log": [_log("Eligibility Check", msg)]}


def fraud_detection_agent(state: ClaimState):
    claim = state["claim"]
    policy = POLICIES.get(claim["policy_number"].strip().upper(), {})
    amount, ctype = claim["amount"], claim["claim_type"]
    typical = TYPICAL_AMOUNT.get(ctype, 150_000)
    indicators = []

    def flag(text, pts):
        indicators.append({"indicator": text, "points": pts})

    # 1. Amount vs typical
    if amount > 3 * typical:
        flag(f"Amount is {amount / typical:.1f}x the typical {ctype} claim (Rs {typical:,})", 25)
    elif amount > 2 * typical:
        flag(f"Amount is {amount / typical:.1f}x the typical {ctype} claim", 12)
    # 2. Amount vs sum insured
    if policy and amount >= 0.7 * policy["sum_insured"]:
        flag(f"Claim uses {amount / policy['sum_insured']:.0%} of the sum insured", 15)
    # 3. New policy
    if policy:
        age = (date.fromisoformat(claim["incident_date"]) - date.fromisoformat(policy["start"])).days
        if 0 <= age < 30:
            flag(f"Incident only {age} days after policy inception", 25)
        # 4. Claim frequency
        prior = policy.get("prior_claims_12m", 0)
        if prior >= 3:
            flag(f"{prior} claims in the last 12 months", 15)
        elif prior == 2:
            flag("2 claims in the last 12 months", 8)
    # 5. Round-number amounts
    if amount >= 100_000 and amount % 50_000 == 0:
        flag("Suspiciously round claim amount", 5)
    # 6. Late reporting
    delay = (date.today() - date.fromisoformat(claim["incident_date"])).days
    if delay > 90:
        flag(f"Claim reported {delay} days after the incident", 10)
    # 7. Description red flags
    text = claim.get("description", "").lower()
    hits = [w for w in SUSPICIOUS_WORDS if w in text]
    if hits:
        flag("Red-flag wording in description: " + ", ".join(hits), min(15, 5 * len(hits)))

    score = min(100, sum(i["points"] for i in indicators))

    # LLM second opinion (adds at most +10, never lowers the score)
    llm_note, llm = "LLM not used (no API key or disabled)", get_llm()
    if llm:
        try:
            opinion = llm.with_structured_output(FraudOpinion).invoke([
                ("system", "You are a conservative insurance fraud analyst. Set concern=true ONLY if the "
                           "claim narrative or the rule-based indicators clearly suggest fraud. "
                           "A routine, coherent claim with no indicators must get concern=false. "
                           "Rationale: max 2 sentences."),
                ("human", json.dumps({"claim": claim, "rule_indicators": indicators, "rule_score": score})),
            ])
            llm_note = opinion.rationale
            if opinion.concern:
                flag("LLM analyst raised a fraud concern", 10)
                score = min(100, score + 10)
        except Exception as exc:  # network / key / quota problems must never crash the workflow
            llm_note = f"LLM unavailable ({type(exc).__name__}); rule-based score only"

    level = "High" if score >= FRAUD_HIGH_AT else "Medium" if score >= 25 else "Low"
    result = {"score": score, "level": level, "indicators": indicators, "llm_note": llm_note}
    return {"fraud_result": result,
            "audit_log": [_log("Fraud Detection", f"score {score}/100 ({level}), {len(indicators)} indicator(s)")]}


def decision_engine(state: ClaimState):
    """Join node: runs after all three parallel checks have finished."""
    claim, docs = state["claim"], state["doc_result"]
    elig, fraud = state["eligibility_result"], state["fraud_result"]
    reasons = []

    if not docs["complete"]:
        decision = "auto_reject"
        reasons.append("Missing required documents: " + ", ".join(docs["missing"]))
    elif not elig["eligible"]:
        decision = "auto_reject"
        reasons += [f"{c['check']} failed - {c['detail']}" for c in elig["checks"] if not c["passed"] and c["hard"]]
    else:
        if claim["amount"] >= HIGH_VALUE_THRESHOLD:
            reasons.append(f"High-value claim (Rs {claim['amount']:,.0f} >= Rs {HIGH_VALUE_THRESHOLD:,})")
        if fraud["score"] >= FRAUD_ESCALATE_AT:
            reasons.append(f"Fraud risk score {fraud['score']}/100 ({fraud['level']})")
        if elig["needs_review"]:
            reasons.append("Claim exceeds remaining coverage - partial payout needs review")
        if reasons:
            decision = "human_review"
        else:
            decision = "auto_approve"
            reasons.append("Documents complete, policy eligible, low fraud risk, below high-value threshold")

    return {"decision": decision, "decision_reasons": reasons,
            "audit_log": [_log("Decision Engine", f"{decision}: {'; '.join(reasons)}")]}


def _template_summary(state: ClaimState) -> str:
    c, d, e, f = state["claim"], state["doc_result"], state["eligibility_result"], state["fraud_result"]
    label = {"auto_approve": "AUTO-APPROVE", "auto_reject": "REJECT", "human_review": "ESCALATE TO HUMAN REVIEW"}
    failed = [x["check"] for x in e["checks"] if not x["passed"]]
    return (
        f"**Claim:** {c['claimant']} - {c['claim_type']} claim of Rs {c['amount']:,.0f} on {c['policy_number']} "
        f"(incident {c['incident_date']}).\n\n"
        f"**Documents:** {len(d['provided'])}/{len(d['required'])} present"
        f"{'' if d['complete'] else ' - missing ' + ', '.join(d['missing'])}.\n\n"
        f"**Eligibility:** {'Eligible' if e['eligible'] else 'Not eligible'}"
        f"{'' if not failed else ' - failed: ' + ', '.join(failed)}.\n\n"
        f"**Fraud risk:** {f['score']}/100 ({f['level']})"
        f"{'' if not f['indicators'] else ' - ' + '; '.join(i['indicator'] for i in f['indicators'])}.\n\n"
        f"**Recommendation:** {label[state['decision']]} - {'; '.join(state['decision_reasons'])}."
    )


def claim_summary_agent(state: ClaimState):
    fallback = _template_summary(state)
    llm, source = get_llm(), "template"
    summary = fallback
    if llm:
        try:
            payload = {k: state[k] for k in
                       ("claim", "doc_result", "eligibility_result", "fraud_result", "decision", "decision_reasons")}
            resp = llm.invoke([
                ("system", "You are an insurance claims analyst. Write a concise summary (max 150 words) for a "
                           "claims manager in markdown with bold labels: Claim, Documents, Eligibility, Fraud risk, "
                           "Recommendation. Use only the facts provided; never change the recommended decision."),
                ("human", json.dumps(payload, default=str)),
            ])
            summary, source = resp.content, "LLM"
        except Exception as exc:
            summary, source = fallback, f"template (LLM error: {type(exc).__name__})"
    return {"summary": summary, "audit_log": [_log("Claim Summary", f"summary generated via {source}")]}


def route_decision(state: ClaimState) -> Literal["auto_approve", "auto_reject", "human_approval"]:
    return {"auto_approve": "auto_approve", "auto_reject": "auto_reject",
            "human_review": "human_approval"}[state["decision"]]


def auto_approve(state: ClaimState):
    return {"final_decision": "APPROVED", "resolved_by": "Automated",
            "audit_log": [_log("Auto Approve", "claim approved automatically")]}


def auto_reject(state: ClaimState):
    return {"final_decision": "REJECTED", "resolved_by": "Automated",
            "audit_log": [_log("Auto Reject", "claim rejected automatically")]}


def human_approval_agent(state: ClaimState):
    """Human-in-the-loop: pauses the graph until a reviewer resumes it."""
    review = interrupt({
        "claim": state["claim"],
        "summary": state["summary"],
        "reasons": state["decision_reasons"],
        "fraud_score": state["fraud_result"]["score"],
    })
    verdict = "APPROVED" if str(review.get("decision", "")).upper().startswith("APPROVE") else "REJECTED"
    return {"final_decision": verdict, "resolved_by": review.get("reviewer", "Human reviewer"),
            "human_review": review,
            "audit_log": [_log("Human Approval", f"reviewer decision: {verdict}"
                                                 f"{' - ' + review['notes'] if review.get('notes') else ''}")]}


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def build_graph():
    g = StateGraph(ClaimState)
    g.add_node("intake", intake)
    g.add_node("document_verification", document_verification_agent)
    g.add_node("eligibility_check", eligibility_check_agent)
    g.add_node("fraud_detection", fraud_detection_agent)
    g.add_node("decision_engine", decision_engine)
    g.add_node("claim_summary", claim_summary_agent)
    g.add_node("auto_approve", auto_approve)
    g.add_node("auto_reject", auto_reject)
    g.add_node("human_approval", human_approval_agent)

    g.add_edge(START, "intake")
    # Parallel fan-out: all three run in the same superstep
    for n in ("document_verification", "eligibility_check", "fraud_detection"):
        g.add_edge("intake", n)
    # Fan-in: decision_engine waits for ALL three
    g.add_edge(["document_verification", "eligibility_check", "fraud_detection"], "decision_engine")
    g.add_edge("decision_engine", "claim_summary")
    # Conditional routing
    g.add_conditional_edges("claim_summary", route_decision,
                            {"auto_approve": "auto_approve", "auto_reject": "auto_reject",
                             "human_approval": "human_approval"})
    for n in ("auto_approve", "auto_reject", "human_approval"):
        g.add_edge(n, END)

    return g.compile(checkpointer=MemorySaver())
