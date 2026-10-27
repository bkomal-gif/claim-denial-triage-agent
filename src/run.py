"""Triage one claim interactively, with a human approval step for appeals.

Usage:
    python src/run.py --claim CLM-1004                 # rules baseline, offline
    python src/run.py --claim CLM-1004 --mode llm      # LLM tool-calling agent
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from langgraph.types import Command

from graph import build_graph

DATA = Path(__file__).resolve().parent.parent / "data" / "claims.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--claim", required=True, help="claim_id, e.g. CLM-1004")
    p.add_argument("--mode", choices=["rules", "llm"], default="rules")
    args = p.parse_args()

    claims = {c["claim_id"]: c for c in json.loads(DATA.read_text())}
    claim = {k: v for k, v in claims[args.claim].items() if k != "expected_action"}

    graph = build_graph()
    config = {"configurable": {"thread_id": args.claim}}
    result = graph.invoke({"claim": claim, "mode": args.mode, "require_approval": True}, config)

    # If the graph paused for human review, show the draft and ask.
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n--- HUMAN REVIEW REQUIRED ---")
        print(f"Claim: {payload['claim_id']}  Decision: {payload['decision']}")
        print(f"Rationale: {payload['rationale']}\n")
        print(payload["appeal_draft"])
        answer = input(payload["question"] + " ")
        result = graph.invoke(Command(resume=answer), config)

    print("\n--- RESULT ---")
    print(f"Decision : {result['decision']}")
    print(f"Rationale: {result['rationale']}")
    print(f"Approved : {result.get('approved')}")
    print(f"Tools    : {' -> '.join(result.get('trace', []))}")


if __name__ == "__main__":
    main()
