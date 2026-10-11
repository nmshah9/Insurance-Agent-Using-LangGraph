import os
import uuid
from datetime import date

import streamlit as st
from langgraph.types import Command

from claims_agent import (ALL_DOC_TYPES, HIGH_VALUE_THRESHOLD, POLICIES, REQUIRED_DOCS,
                          build_graph, infer_docs_from_filenames)
from scenarios import SCENARIOS

st.set_page_config(page_title="Insurance Claim Agent", page_icon="🛡️", layout="wide")


@st.cache_resource
def get_graph():
    return build_graph()


graph = get_graph()

# ----------------------------- session defaults ----------------------------- #
DEFAULTS = {"policy_number": "POL-1001", "claimant": "Asha Mehta", "claim_type": "Health",
            "incident_date": date.today(), "amount": 45_000, "description": "",
            "documents": REQUIRED_DOCS["Health"]}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def load_scenario(i: int):
    for k, v in SCENARIOS[i]["claim"].items():
        st.session_state[k] = v
    if "upload_folder" in SCENARIOS[i]:          # upload-test scenarios: documents come from the uploader
        st.session_state["documents"] = []
        st.session_state["upload_hint"] = SCENARIOS[i]["upload_folder"]
    else:
        st.session_state.pop("upload_hint", None)
    st.session_state.pop("thread_id", None)


# --------------------------------- sidebar ---------------------------------- #
with st.sidebar:
    st.header("Test scenarios")
    for i, sc in enumerate(SCENARIOS):
        st.button(sc["name"], key=f"sc{i}", on_click=load_scenario, args=(i,), use_container_width=True)

    st.divider()
    st.header("LLM settings")
    secret_key = ""
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        pass
    api_key = st.text_input("OpenAI API key", type="password", value=secret_key or os.getenv("OPENAI_API_KEY", ""),
                            help="Optional. Without a key the agents use rule-based fallbacks.")
    use_llm = st.toggle("Use LLM (GPT-4o-mini)", value=bool(api_key))
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    os.environ["CLAIMS_USE_LLM"] = "1" if (use_llm and api_key) else "0"
    st.caption("LLM mode: **ON**" if (use_llm and api_key) else "LLM mode: **OFF** (rule-based fallback)")

    with st.expander("Mock policy database"):
        st.json({k: {**v, "holder": v["holder"]} for k, v in POLICIES.items()})

# ----------------------------------- form ----------------------------------- #
st.title("🛡️ Insurance Claim Processing Agent")
st.caption("LangGraph · parallel verification / eligibility / fraud checks · conditional routing · human-in-the-loop")

c1, c2, c3 = st.columns(3)
c1.text_input("Policy number", key="policy_number")
c2.text_input("Claimant name", key="claimant")
c3.selectbox("Claim type", list(REQUIRED_DOCS), key="claim_type")
c4, c5 = st.columns(2)
c4.date_input("Incident date", key="incident_date")
c5.number_input("Claim amount (Rs)", min_value=0, step=1000, key="amount")
st.text_area("Incident description", key="description", height=80)

st.multiselect("Documents submitted", ALL_DOC_TYPES, key="documents",
               help=f"Required for {st.session_state.claim_type}: {', '.join(REQUIRED_DOCS[st.session_state.claim_type])}")
if st.session_state.get("upload_hint"):
    st.info(f"Upload the files from `{st.session_state['upload_hint']}` below (select all of them).")
uploads = st.file_uploader("...or upload supporting files (type inferred from file name, e.g. hospital_bill.pdf)",
                           accept_multiple_files=True)

if st.button("🚀 Process claim", type="primary"):
    docs = sorted(set(st.session_state.documents) | set(infer_docs_from_filenames([u.name for u in uploads])))
    claim = {
        "policy_number": st.session_state.policy_number, "claimant": st.session_state.claimant,
        "claim_type": st.session_state.claim_type,
        "incident_date": st.session_state.incident_date.isoformat(),
        "amount": float(st.session_state.amount), "description": st.session_state.description,
        "documents": docs,
    }
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    with st.status("Running claim workflow...", expanded=True) as status:
        for chunk in graph.stream({"claim": claim, "audit_log": []}, cfg, stream_mode="updates"):
            for node in chunk:
                status.write("⏸️ Waiting for human reviewer" if node == "__interrupt__" else f"✅ {node}")
        status.update(label="Workflow finished", state="complete")
    st.session_state["thread_id"] = thread_id

# --------------------------------- results ---------------------------------- #
if "thread_id" in st.session_state:
    cfg = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    snap = graph.get_state(cfg)
    v, pending = snap.values, bool(snap.next)

    st.divider()
    st.subheader("Result")
    if pending:
        st.warning("⏸️ **Escalated for human review** - the workflow is paused until a reviewer decides.")
    elif v.get("final_decision") == "APPROVED":
        st.success(f"✅ **Claim APPROVED** ({v['resolved_by']})")
    else:
        st.error(f"❌ **Claim REJECTED** ({v['resolved_by']})")

    d, e, f = v["doc_result"], v["eligibility_result"], v["fraud_result"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Documents", f"{len(d['provided'])}/{len(d['required'])}")
    m2.metric("Eligibility", "Eligible" if e["eligible"] else "Not eligible")
    m3.metric("Fraud score", f"{f['score']}/100", f["level"], delta_color="off")
    m4.metric("Routing", v["decision"].replace("_", " ").title())

    st.markdown("**Why:** " + " · ".join(v["decision_reasons"]))

    t1, t2, t3, t4 = st.tabs(["📝 Claim summary", "Parallel checks", "Audit log", "Workflow graph"])
    with t1:
        st.markdown(v["summary"])
    with t2:
        a, b, c = st.columns(3)
        with a:
            st.markdown("##### Document Verification")
            for doc in d["required"]:
                st.write(("✅ " if doc in d["provided"] else "❌ ") + doc)
        with b:
            st.markdown("##### Eligibility Check")
            for ch in e["checks"]:
                st.write(("✅ " if ch["passed"] else "❌ ") + f"{ch['check']} - {ch['detail']}")
        with c:
            st.markdown("##### Fraud Detection")
            if not f["indicators"]:
                st.write("No fraud indicators found")
            for ind in f["indicators"]:
                st.write(f"⚠️ {ind['indicator']} (+{ind['points']})")
            st.caption(f"LLM note: {f['llm_note']}")
    with t3:
        st.code("\n".join(v["audit_log"]), language="text")
    with t4:
        st.graphviz_chart("""
        digraph { rankdir=LR; node [shape=box, style=rounded];
          intake -> document_verification; intake -> eligibility_check; intake -> fraud_detection;
          document_verification -> decision_engine; eligibility_check -> decision_engine;
          fraud_detection -> decision_engine; decision_engine -> claim_summary;
          claim_summary -> auto_approve [label="approve"]; claim_summary -> auto_reject [label="reject"];
          claim_summary -> human_approval [label="escalate"]; }
        """)

    # ----------------------------- human-in-the-loop ----------------------------- #
    if pending:
        st.divider()
        st.subheader("👤 Human Approval")
        st.info(f"Escalation triggers: {' · '.join(v['decision_reasons'])}")
        notes = st.text_area("Reviewer notes", key="reviewer_notes")
        ca, cb = st.columns(2)
        decision = None
        if ca.button("✅ Approve claim", use_container_width=True):
            decision = "APPROVE"
        if cb.button("❌ Reject claim", use_container_width=True):
            decision = "REJECT"
        if decision:
            graph.invoke(Command(resume={"decision": decision, "notes": notes, "reviewer": "Claims Manager"}), cfg)
            st.rerun()
