"""Tools the triage agent can call.

Each tool is a LangChain `@tool`, so the same functions work in both modes:
- rules mode: graph nodes call them directly (deterministic baseline)
- llm mode: the model chooses which tools to call (tool-calling agent)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.tools import tool

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Subset of CARC codes covered in this project, with plain-language meaning
# and the usual category of fix.
CARC_CODES = {
    "CO-4": {"meaning": "Procedure code inconsistent with modifier, or required modifier missing", "category": "coding_error"},
    "CO-16": {"meaning": "Claim/service lacks information or has submission/billing errors", "category": "missing_information"},
    "CO-18": {"meaning": "Exact duplicate claim/service", "category": "duplicate"},
    "CO-29": {"meaning": "The time limit for filing has expired", "category": "timely_filing"},
    "CO-50": {"meaning": "Non-covered: not deemed a medical necessity by the payer", "category": "medical_necessity"},
    "CO-97": {"meaning": "Benefit included in the payment/allowance for another service (bundled)", "category": "bundling"},
    "CO-197": {"meaning": "Precertification/authorization/notification absent", "category": "authorization"},
}


def _load_policy_sections() -> list[tuple[str, str]]:
    text = (DATA_DIR / "payer_policies.md").read_text()
    sections = re.split(r"\n## ", text)[1:]
    out = []
    for s in sections:
        title, _, body = s.partition("\n")
        out.append((title.strip(), body.strip()))
    return out


POLICY_SECTIONS = _load_policy_sections()


@tool
def lookup_denial_code(code: str) -> str:
    """Look up a CARC denial code and return its meaning and category as JSON."""
    info = CARC_CODES.get(code.strip().upper())
    if info is None:
        return json.dumps({"code": code, "error": "unknown code - escalate to a human analyst"})
    return json.dumps({"code": code, **info})


@tool
def check_claim_fields(claim_json: str) -> str:
    """Check a claim (as JSON) for the facts that decide the next action:
    missing fields, duplicate status, prior auth, timely-filing proof,
    clinical documentation and distinct-service documentation."""
    c = json.loads(claim_json)
    findings = {
        "missing_fields": c.get("missing_fields", []),
        "is_duplicate_of_paid_claim": c.get("is_duplicate_of_paid_claim", False),
        "prior_auth_on_file": c.get("prior_auth_on_file"),
        "proof_of_timely_filing": c.get("proof_of_timely_filing"),
        "clinical_documentation_available": c.get("clinical_documentation_available"),
        "distinct_service_documented": c.get("distinct_service_documented"),
    }
    return json.dumps(findings)


@tool
def search_payer_policy(query: str) -> str:
    """Search payer policy excerpts and return the most relevant section.
    Simple keyword-overlap retrieval; swap for FAISS/embeddings at scale."""
    q = set(re.findall(r"[a-z0-9-]+", query.lower()))
    best, best_score = None, -1
    for title, body in POLICY_SECTIONS:
        words = set(re.findall(r"[a-z0-9-]+", (title + " " + body).lower()))
        score = len(q & words) + (5 if any(tok.upper() in title for tok in q) else 0)
        if score > best_score:
            best, best_score = (title, body), score
    return f"{best[0]}\n{best[1]}"


@tool
def draft_appeal(claim_json: str, reason: str) -> str:
    """Draft a short appeal letter for a denied claim, given the claim JSON
    and the reason the denial should be overturned."""
    c = json.loads(claim_json)
    return (
        f"RE: Appeal of denied claim {c['claim_id']} ({c['denial_code']})\n"
        f"Payer: {c['payer']}\n"
        f"Service: CPT {c['cpt_code']} / ICD-10 {c['icd10_code']} - {c['service_description']}\n"
        f"Date of service: {c['date_of_service']}\n\n"
        f"We request reconsideration of this denial. {reason}\n"
        "Supporting documentation is enclosed. Please reprocess the claim for payment.\n"
    )


ALL_TOOLS = [lookup_denial_code, check_claim_fields, search_payer_policy, draft_appeal]
