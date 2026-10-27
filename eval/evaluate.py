"""Evaluate the triage agent on the labelled synthetic claims.

Metrics
- decision accuracy (overall, with a 95% bootstrap confidence interval)
- per-denial-code accuracy
- confusion matrix (expected vs predicted action)
- tool-call recall: did the agent call each tool a careful analyst would use?
- escalation rate

Usage:
    python eval/evaluate.py                  # rules baseline (offline)
    python eval/evaluate.py --mode llm       # LLM agent (needs an API key)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from graph import MEMORY, build_graph  # noqa: E402

REQUIRED_TOOLS = {"lookup_denial_code", "check_claim_fields", "search_payer_policy"}
ACTIONS = ["correct_and_resubmit", "appeal", "write_off", "escalate"]


def bootstrap_ci(hits: list[int], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(hits, k=len(hits))) / len(hits) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["rules", "llm"], default="rules")
    args = p.parse_args()

    claims = json.loads((ROOT / "data" / "claims.json").read_text())
    graph = build_graph()
    MEMORY.reset()

    rows, hits = [], []
    per_code = defaultdict(list)
    confusion = Counter()
    tool_recall = []

    for c in claims:
        expected = c["expected_action"]
        claim = {k: v for k, v in c.items() if k != "expected_action"}
        out = graph.invoke(
            {"claim": claim, "mode": args.mode, "require_approval": False},
            {"configurable": {"thread_id": f"eval-{args.mode}-{c['claim_id']}"}},
        )
        pred = out.get("decision", "escalate")
        hit = int(pred == expected)
        hits.append(hit)
        per_code[c["denial_code"]].append(hit)
        confusion[(expected, pred)] += 1
        called = set(out.get("trace", []))
        tool_recall.append(len(called & REQUIRED_TOOLS) / len(REQUIRED_TOOLS))
        rows.append({"claim_id": c["claim_id"], "denial_code": c["denial_code"], "expected": expected,
                     "predicted": pred, "correct": bool(hit), "tools": out.get("trace", []),
                     "rationale": out.get("rationale", "")})

    acc = sum(hits) / len(hits)
    lo, hi = bootstrap_ci(hits)
    report = [f"# Evaluation report - mode: {args.mode}", "",
              f"Claims evaluated: {len(hits)}",
              f"Decision accuracy: {acc:.1%} (95% bootstrap CI {lo:.1%} - {hi:.1%})",
              f"Mean tool-call recall: {sum(tool_recall) / len(tool_recall):.1%}",
              f"Escalation rate: {sum(r['predicted'] == 'escalate' for r in rows) / len(rows):.1%}", "",
              "## Accuracy by denial code", "", "| Code | n | Accuracy |", "|---|---|---|"]
    for code in sorted(per_code):
        h = per_code[code]
        report.append(f"| {code} | {len(h)} | {sum(h) / len(h):.0%} |")
    report += ["", "## Confusion matrix (rows = expected, columns = predicted)", "",
               "| expected \\ predicted | " + " | ".join(ACTIONS) + " |",
               "|---" * (len(ACTIONS) + 1) + "|"]
    for e in ACTIONS:
        report.append(f"| {e} | " + " | ".join(str(confusion[(e, p_)]) for p_ in ACTIONS) + " |")
    errors = [r for r in rows if not r["correct"]]
    report += ["", f"## Errors ({len(errors)})", ""]
    report += [f"- {r['claim_id']} ({r['denial_code']}): expected {r['expected']}, got {r['predicted']} - {r['rationale']}"
               for r in errors] or ["- none"]

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"eval_{args.mode}.md").write_text("\n".join(report) + "\n")
    (out_dir / f"predictions_{args.mode}.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(report))


if __name__ == "__main__":
    main()
