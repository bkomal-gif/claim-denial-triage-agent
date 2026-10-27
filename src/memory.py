"""Long-term case memory across claims.

LangGraph's checkpointer gives short-term memory within one claim's run
(and lets a run pause for human review and resume). This module adds simple
long-term memory across runs: how often each (payer, denial code) pair ended
in each action, so the agent can see patterns such as "Payer B usually
denies CO-197 without auth on file".
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "results" / "case_memory.json"


class CaseMemory:
    def __init__(self, path: Path | None = None, persist: bool = False):
        self.path = path or DEFAULT_PATH
        self.persist = persist
        self.data: dict[str, Counter] = {}
        if persist and self.path.exists():
            raw = json.loads(self.path.read_text())
            self.data = {k: Counter(v) for k, v in raw.items()}

    @staticmethod
    def _key(payer: str, code: str) -> str:
        return f"{payer}|{code}"

    def record(self, payer: str, code: str, action: str) -> None:
        self.data.setdefault(self._key(payer, code), Counter())[action] += 1
        if self.persist:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=2))

    def summary(self, payer: str, code: str) -> str:
        counts = self.data.get(self._key(payer, code))
        if not counts:
            return "no previous cases"
        parts = ", ".join(f"{a}: {n}" for a, n in counts.most_common())
        return f"{sum(counts.values())} previous cases ({parts})"

    def reset(self) -> None:
        self.data = {}
