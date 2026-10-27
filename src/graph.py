"""LangGraph workflow for claim-denial triage.

Flow:
    intake -> (rules: gather -> decide) | (llm: agent <-> tools -> parse)
           -> [appeal?] draft -> human_review (interrupt) -> finalize
           -> [else]  finalize

Two modes share the same tools, state and human-review step:
- "rules": deterministic baseline, runs offline with no API key
- "llm":   a tool-calling LLM decides which tools to call and makes the decision
"""
from __future__ import annotations

import json
import os
import re
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from memory import CaseMemory
from tools import (
    ALL_TOOLS,
    check_claim_fields,
    draft_appeal,
    lookup_denial_code,
    search_payer_policy,
)

Action = Literal["correct_and_resubmit", "appeal", "write_off", "escalate"]
VALID_ACTIONS = {"correct_and_resubmit", "appeal", "write_off", "escalate"}


class TriageState(TypedDict, total=False):
    claim: dict
    mode: str
    require_approval: bool
    memory_note: str
    denial_info: dict
    findings: dict
    policy: str
    decision: Action
    rationale: str
    appeal_draft: Optional[str]
    approved: Optional[bool]
    trace: list[str]  # tools called, in order (used for evaluation)
    messages: Annotated[list[AnyMessage], add_messages]


MEMORY = CaseMemory()

# ---------------------------------------------------------------- shared nodes


def intake(state: TriageState) -> dict:
    claim = state["claim"]
    required = {"claim_id", "denial_code", "payer", "cpt_code", "icd10_code"}
    missing = required - claim.keys()
    if missing:
        return {"decision": "escalate", "rationale": f"Malformed claim, missing {sorted(missing)}", "trace": []}
    note = MEMORY.summary(claim["payer"], claim["denial_code"])
    return {"memory_note": note, "trace": []}


def route_after_intake(state: TriageState) -> str:
    if state.get("decision") == "escalate":
        return "finalize"
    return "gather" if state.get("mode", "rules") == "rules" else "agent"


def draft(state: TriageState) -> dict:
    letter = draft_appeal.invoke({"claim_json": json.dumps(state["claim"]), "reason": state["rationale"]})
    return {"appeal_draft": letter, "trace": state["trace"] + ["draft_appeal"]}


def human_review(state: TriageState) -> dict:
    """Pause for a human to approve the appeal before anything is sent."""
    if not state.get("require_approval", True):
        return {"approved": True}
    answer = interrupt(
        {
            "claim_id": state["claim"]["claim_id"],
            "decision": state["decision"],
            "rationale": state["rationale"],
            "appeal_draft": state.get("appeal_draft"),
            "question": "Approve this appeal? (yes/no)",
        }
    )
    return {"approved": str(answer).strip().lower() in {"y", "yes", "approve", "true"}}


def finalize(state: TriageState) -> dict:
    c = state["claim"]
    if "denial_code" in c and "payer" in c:
        MEMORY.record(c["payer"], c["denial_code"], state.get("decision", "escalate"))
    return {}


def route_after_decision(state: TriageState) -> str:
    return "draft" if state.get("decision") == "appeal" else "finalize"


# ------------------------------------------------------------ rules baseline


def gather(state: TriageState) -> dict:
    c = state["claim"]
    info = json.loads(lookup_denial_code.invoke({"code": c["denial_code"]}))
    findings = json.loads(check_claim_fields.invoke({"claim_json": json.dumps(c)}))
    policy = search_payer_policy.invoke({"query": f"{c['denial_code']} {info.get('meaning', '')}"})
    return {
        "denial_info": info,
        "findings": findings,
        "policy": policy,
        "trace": state["trace"] + ["lookup_denial_code", "check_claim_fields", "search_payer_policy"],
    }


def decide_rules(state: TriageState) -> dict:
    code, f = state["claim"]["denial_code"], state["findings"]
    if code in {"CO-16", "CO-4"}:
        return {"decision": "correct_and_resubmit", "rationale": f"Add missing information: {f['missing_fields']}."}
    if code == "CO-18":
        return {"decision": "write_off", "rationale": "Duplicate of an adjudicated claim; no action needed."}
    rules = {
        "CO-29": ("proof_of_timely_filing", "Proof of timely filing is on file.", "No proof of timely filing."),
        "CO-50": ("clinical_documentation_available", "Clinical documentation supports medical necessity.", "No supporting clinical documentation."),
        "CO-197": ("prior_auth_on_file", "Prior authorization was obtained; attaching the authorization number.", "No authorization on file."),
        "CO-97": ("distinct_service_documented", "Service was separate and distinct; resubmitting with modifier 59.", "Service was not distinct; bundling is correct."),
    }
    if code in rules:
        key, yes, no = rules[code]
        return {"decision": "appeal", "rationale": yes} if f.get(key) else {"decision": "write_off", "rationale": no}
    return {"decision": "escalate", "rationale": f"No rule for {code}."}


# --------------------------------------------------------------- LLM agent

SYSTEM_PROMPT = """You are a healthcare revenue-cycle analyst triaging a denied medical claim.
Use the tools to (1) look up the denial code, (2) check the claim's fields, and
(3) search payer policy. Do not guess facts that a tool can check.
Then choose exactly one action:
- correct_and_resubmit: fixable data or coding errors (missing info, missing modifier)
- appeal: the denial is wrong AND the claim has the evidence the policy requires
- write_off: the denial is correct or the required evidence is missing
- escalate: unclear or unknown situation
Do not call draft_appeal; drafting happens later after human review.
Finish with ONLY a JSON object: {"decision": "...", "rationale": "..."}"""


def _get_model():
    from langchain.chat_models import init_chat_model  # optional dependency

    model_name = os.getenv("TRIAGE_MODEL", "openai:gpt-4o-mini")
    tools = [t for t in ALL_TOOLS if t.name != "draft_appeal"]
    return init_chat_model(model_name, temperature=0).bind_tools(tools)


def agent(state: TriageState) -> dict:
    msgs = state.get("messages") or [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(
            "Claim:\n" + json.dumps(state["claim"]) + "\n\nPast cases for this payer and code: " + state.get("memory_note", "none")
        ),
    ]
    response = _get_model().invoke(msgs)
    new = msgs + [response] if not state.get("messages") else [response]
    calls = [tc["name"] for tc in getattr(response, "tool_calls", [])]
    return {"messages": new, "trace": state["trace"] + calls}


def route_agent(state: TriageState) -> str:
    last = state["messages"][-1]
    steps = sum(isinstance(m, AIMessage) for m in state["messages"])
    if isinstance(last, AIMessage) and last.tool_calls and steps < 8:
        return "tools"
    return "parse"


def parse_decision(state: TriageState) -> dict:
    text = state["messages"][-1].content
    if isinstance(text, list):  # some providers return content blocks
        text = " ".join(b.get("text", "") for b in text if isinstance(b, dict))
    m = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    decision = data.get("decision", "escalate")
    if decision not in VALID_ACTIONS:
        decision = "escalate"
    return {"decision": decision, "rationale": data.get("rationale", "Could not parse model output.")}


# ------------------------------------------------------------------- build


def build_graph(checkpointer=None):
    g = StateGraph(TriageState)
    g.add_node("intake", intake)
    g.add_node("gather", gather)
    g.add_node("decide_rules", decide_rules)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode([t for t in ALL_TOOLS if t.name != "draft_appeal"]))
    g.add_node("parse", parse_decision)
    g.add_node("draft", draft)
    g.add_node("human_review", human_review)
    g.add_node("finalize", finalize)

    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", route_after_intake, ["gather", "agent", "finalize"])
    g.add_edge("gather", "decide_rules")
    g.add_conditional_edges("decide_rules", route_after_decision, ["draft", "finalize"])
    g.add_conditional_edges("agent", route_agent, ["tools", "parse"])
    g.add_edge("tools", "agent")
    g.add_conditional_edges("parse", route_after_decision, ["draft", "finalize"])
    g.add_edge("draft", "human_review")
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
