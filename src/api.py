"""REST API for the claim-denial triage agent.

Endpoints
- GET  /health                       liveness check
- GET  /claims                       list the sample claim IDs
- GET  /claims/{claim_id}            one sample claim
- POST /triage                       triage a claim; appeals pause for approval
- POST /triage/{thread_id}/review    approve or reject a paused appeal

Run locally:  uvicorn api:app --app-dir src --port 7860
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from graph import build_graph

DATA = Path(__file__).resolve().parent.parent / "data" / "claims.json"
SAMPLES = {c["claim_id"]: {k: v for k, v in c.items() if k != "expected_action"}
           for c in json.loads(DATA.read_text())}

app = FastAPI(
    title="Claim Denial Triage Agent",
    description="LangGraph agent that triages denied healthcare claims (synthetic data only), "
                "with a human approval step for appeals.",
    version="0.2.0",
)
GRAPH = build_graph()  # one compiled graph; the checkpointer keeps each thread's state


class TriageRequest(BaseModel):
    claim_id: Optional[str] = Field(None, description="Use a sample claim, e.g. CLM-1004")
    claim: Optional[dict] = Field(None, description="Or send your own claim object")
    mode: Literal["rules", "llm"] = "rules"


class ReviewRequest(BaseModel):
    approve: bool


def _summary(thread_id: str, out: dict) -> dict:
    if "__interrupt__" in out:
        pending = out["__interrupt__"][0].value
        return {"thread_id": thread_id, "status": "pending_approval", "decision": pending["decision"],
                "rationale": pending["rationale"], "appeal_draft": pending["appeal_draft"],
                "next": f"POST /triage/{thread_id}/review with {{\"approve\": true|false}}"}
    return {"thread_id": thread_id, "status": "completed", "decision": out.get("decision"),
            "rationale": out.get("rationale"), "approved": out.get("approved"),
            "appeal_draft": out.get("appeal_draft"), "tools_called": out.get("trace", [])}


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok", "llm_mode_available": bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))}


@app.get("/claims")
def list_claims():
    return {"claim_ids": list(SAMPLES)}


@app.get("/claims/{claim_id}")
def get_claim(claim_id: str):
    if claim_id not in SAMPLES:
        raise HTTPException(404, f"Unknown claim {claim_id}")
    return SAMPLES[claim_id]


@app.post("/triage")
def triage(req: TriageRequest):
    if req.claim is None and req.claim_id is None:
        raise HTTPException(422, "Send either claim_id or claim")
    claim = req.claim or SAMPLES.get(req.claim_id)
    if claim is None:
        raise HTTPException(404, f"Unknown claim {req.claim_id}")
    thread_id = f"{claim.get('claim_id', 'claim')}-{uuid.uuid4().hex[:8]}"
    try:
        out = GRAPH.invoke({"claim": claim, "mode": req.mode, "require_approval": True},
                           {"configurable": {"thread_id": thread_id}})
    except Exception as e:  # e.g. llm mode without an API key
        raise HTTPException(500, f"Triage failed: {type(e).__name__}: {e}")
    return _summary(thread_id, out)


@app.post("/triage/{thread_id}/review")
def review(thread_id: str, req: ReviewRequest):
    config = {"configurable": {"thread_id": thread_id}}
    state = GRAPH.get_state(config)
    if not state.next:
        raise HTTPException(409, "Nothing is waiting for review on this thread")
    out = GRAPH.invoke(Command(resume="yes" if req.approve else "no"), config)
    return _summary(thread_id, out)
