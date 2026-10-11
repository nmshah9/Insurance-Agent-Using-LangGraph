"""Checkpoint 1: run all 5 scenarios headlessly (no Streamlit, no API key needed).

    python test_scenarios.py
"""
import uuid

from langgraph.types import Command

from claims_agent import build_graph
from scenarios import SCENARIOS


def run(graph, scenario):
    claim = dict(scenario["claim"])
    claim["incident_date"] = claim["incident_date"].isoformat()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    graph.invoke({"claim": claim, "audit_log": []}, cfg)
    snap = graph.get_state(cfg)
    paused = bool(snap.next)
    if paused:  # human-in-the-loop: simulate the reviewer approving
        graph.invoke(Command(resume={"decision": "APPROVE", "notes": "test run", "reviewer": "tester"}), cfg)
        snap = graph.get_state(cfg)
    return snap.values, paused


if __name__ == "__main__":
    graph = build_graph()
    ok = True
    for sc in SCENARIOS:
        values, paused = run(graph, sc)
        passed = values["decision"] == sc["expected"]
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  {sc['name']}")
        print(f"      route={values['decision']}  paused_for_human={paused}  final={values['final_decision']} "
              f"({values['resolved_by']})  fraud={values['fraud_result']['score']}")
        for r in values["decision_reasons"]:
            print(f"      - {r}")
    print("\nALL SCENARIOS PASSED" if ok else "\nSOME SCENARIOS FAILED")
    raise SystemExit(0 if ok else 1)
