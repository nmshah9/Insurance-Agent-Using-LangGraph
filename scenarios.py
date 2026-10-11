"""Five demo claim scenarios. Dates are relative to today so they never go stale."""
from datetime import date, timedelta

from claims_agent import REQUIRED_DOCS


def _ago(days: int) -> date:
    return date.today() - timedelta(days=days)


SCENARIOS = [
    {
        "name": "1. Complete claim -> Auto Approve",
        "expected": "auto_approve",
        "claim": {
            "policy_number": "POL-1001", "claimant": "Asha Mehta", "claim_type": "Health",
            "incident_date": _ago(20), "amount": 45_000,
            "description": "Hospitalised for appendicitis surgery; admitted for 3 days.",
            "documents": REQUIRED_DOCS["Health"],
        },
    },
    {
        "name": "2. Missing documents -> Reject",
        "expected": "auto_reject",
        "claim": {
            "policy_number": "POL-2002", "claimant": "Rohan Iyer", "claim_type": "Motor",
            "incident_date": _ago(10), "amount": 60_000,
            "description": "Rear bumper damaged in a parking-lot collision.",
            "documents": ["Claim Form", "Driving Licence", "Vehicle RC"],   # no FIR / estimate
        },
    },
    {
        "name": "3. Suspicious amount -> Human Approval",
        "expected": "human_review",
        "claim": {
            "policy_number": "POL-5005", "claimant": "Sanjay Rao", "claim_type": "Motor",
            "incident_date": _ago(8), "amount": 640_000,
            "description": "Vehicle declared a total loss after an accident with no witnesses. "
                           "Need urgent settlement.",
            "documents": REQUIRED_DOCS["Motor"],
        },
    },
    {
        "name": "4. Expired policy -> Reject",
        "expected": "auto_reject",
        "claim": {
            "policy_number": "POL-3003", "claimant": "Kiran Patel", "claim_type": "Health",
            "incident_date": _ago(15), "amount": 80_000,
            "description": "Treatment for a fractured wrist.",
            "documents": REQUIRED_DOCS["Health"],
        },
    },
    {
        "name": "5. High-value, valid docs -> Human Approval",
        "expected": "human_review",
        "claim": {
            "policy_number": "POL-4004", "claimant": "Meera Nair", "claim_type": "Health",
            "incident_date": _ago(12), "amount": 1_200_000,
            "description": "Planned cardiac bypass surgery with 9 days of hospitalisation.",
            "documents": REQUIRED_DOCS["Health"],
        },
    },
    {
        "name": "6. Property claim via uploads -> Auto Approve",
        "expected": "auto_approve",
        "upload_folder": "sample_documents/variant_6_property_fire",
        "claim": {
            "policy_number": "POL-6006", "claimant": "Vikram Desai", "claim_type": "Property",
            "incident_date": _ago(14), "amount": 180_000,
            "description": "Kitchen damaged by an electrical short-circuit fire; contained by the fire brigade.",
            "documents": REQUIRED_DOCS["Property"],
        },
    },
    {
        "name": "7. Exceeds remaining coverage -> Human Approval",
        "expected": "human_review",
        "upload_folder": "sample_documents/variant_7_health_over_limit",
        "claim": {
            "policy_number": "POL-7007", "claimant": "Neha Kulkarni", "claim_type": "Health",
            "incident_date": _ago(9), "amount": 150_000,
            "description": "Emergency admission for pneumonia; 6 days in hospital.",
            "documents": REQUIRED_DOCS["Health"],
        },
    },
]
