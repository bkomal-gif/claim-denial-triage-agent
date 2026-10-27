"""Generate a synthetic dataset of denied medical claims.

All data is SYNTHETIC. No real patient, provider or payer data is used.
Each claim carries a CARC (Claim Adjustment Reason Code) denial code plus the
facts an analyst would check, and an `expected_action` label used for evaluation.
"""
import json
import random
from pathlib import Path

random.seed(42)

PAYERS = ["Payer A (Commercial)", "Payer B (Medicare Advantage)", "Payer C (Medicaid MCO)"]

# (CPT procedure, ICD-10 diagnosis, short description) - realistic code pairs
SERVICES = [
    ("99214", "E11.9", "Office visit, established patient - type 2 diabetes"),
    ("73721", "M23.211", "MRI knee without contrast - meniscus tear"),
    ("29881", "M23.211", "Knee arthroscopy with meniscectomy"),
    ("97110", "M54.50", "Therapeutic exercise - low back pain"),
    ("20610", "M17.11", "Major joint injection - knee osteoarthritis"),
    ("93000", "R07.9", "Electrocardiogram - chest pain"),
    ("72148", "M51.26", "MRI lumbar spine - disc displacement"),
]


def make_claim(i: int, carc: str) -> dict:
    cpt, icd, desc = random.choice(SERVICES)
    claim = {
        "claim_id": f"CLM-{1000 + i}",
        "payer": random.choice(PAYERS),
        "cpt_code": cpt,
        "icd10_code": icd,
        "service_description": desc,
        "modifiers": [],
        "billed_amount_usd": random.choice([180, 420, 950, 1600, 2400]),
        "date_of_service": f"2026-0{random.randint(1, 8)}-{random.randint(10, 28)}",
        "denial_code": carc,
        "missing_fields": [],
        "prior_auth_on_file": None,
        "proof_of_timely_filing": None,
        "clinical_documentation_available": None,
        "distinct_service_documented": None,
        "is_duplicate_of_paid_claim": False,
    }

    if carc == "CO-16":  # claim lacks information -> fix and resubmit
        claim["missing_fields"] = random.choice([["rendering_provider_npi"], ["referring_provider"], ["patient_dob"]])
        expected = "correct_and_resubmit"
    elif carc == "CO-4":  # modifier inconsistent / missing -> fix and resubmit
        claim["cpt_code"], claim["service_description"] = "20610", "Major joint injection - knee osteoarthritis"
        claim["missing_fields"] = ["modifier_RT_or_LT"]
        expected = "correct_and_resubmit"
    elif carc == "CO-18":  # exact duplicate -> no action
        claim["is_duplicate_of_paid_claim"] = True
        expected = "write_off"
    elif carc == "CO-29":  # timely filing limit expired
        has_proof = random.random() < 0.5
        claim["proof_of_timely_filing"] = has_proof
        expected = "appeal" if has_proof else "write_off"
    elif carc == "CO-50":  # not medically necessary
        has_docs = random.random() < 0.6
        claim["clinical_documentation_available"] = has_docs
        expected = "appeal" if has_docs else "write_off"
    elif carc == "CO-197":  # precertification / authorization absent
        has_auth = random.random() < 0.5
        claim["prior_auth_on_file"] = has_auth
        expected = "appeal" if has_auth else "write_off"
    elif carc == "CO-97":  # bundled into another service
        distinct = random.random() < 0.5
        claim["distinct_service_documented"] = distinct
        expected = "appeal" if distinct else "write_off"
    elif carc == "CO-11":  # not in the agent's code table -> should escalate
        expected = "escalate"
    elif carc.startswith("UNK-"):  # known code, but the deciding fact is unknown (None)
        claim["denial_code"] = carc[4:]
        expected = "escalate"
    else:
        raise ValueError(carc)

    claim["expected_action"] = expected
    return claim


def main() -> None:
    codes = ["CO-16", "CO-4", "CO-18", "CO-29", "CO-50", "CO-197", "CO-97"]
    plan = codes * 4 + ["CO-50", "CO-197"]  # 30 standard claims
    # 6 hard cases: an unsupported code, and claims where the deciding fact is unknown
    plan += ["CO-11", "CO-11", "UNK-CO-50", "UNK-CO-197", "UNK-CO-29", "UNK-CO-97"]
    claims = [make_claim(i, c) for i, c in enumerate(plan)]
    out = Path(__file__).parent / "claims.json"
    out.write_text(json.dumps(claims, indent=2))
    counts = {}
    for c in claims:
        counts[c["expected_action"]] = counts.get(c["expected_action"], 0) + 1
    print(f"Wrote {len(claims)} synthetic claims to {out}; label counts: {counts}")


if __name__ == "__main__":
    main()
