"""Stagnation: fire after N evals without a new best version.

State is {"best": version id or None, "stagnant": evals since it changed},
persisted in the run's progress notes so resume keeps the count. After a
fire the counter resets, so a still-stuck run refires once per window rather
than every turn. Threshold 0 disables the trigger.
"""

from __future__ import annotations

from typing import Any


def is_eval_result(tool_name: str | None, result: Any) -> bool:
    """True when a tool result is a completed eval (only run_eval counts).

    read_trace also reports eval_status, but re-reading a trace is not a new
    eval and must not advance the counter.
    """
    return (
        tool_name == "run_eval"
        and isinstance(result, dict)
        and "eval_status" in result
    )


def update(
    state: dict[str, Any] | None, best_id: str | None, threshold: int
) -> tuple[dict[str, Any], bool]:
    """Fold one completed eval into the stagnation state.

    Returns (new state, fired). A changed best (including the first one)
    resets the counter; an unchanged best counts up and fires at threshold.
    """
    current = {"best": None, "stagnant": 0}
    if isinstance(state, dict):
        current["best"] = state.get("best")
        try:
            current["stagnant"] = int(state.get("stagnant", 0) or 0)
        except (TypeError, ValueError):
            current["stagnant"] = 0
    if threshold <= 0:
        return current, False
    if best_id != current["best"]:
        return {"best": best_id, "stagnant": 0}, False
    stagnant = current["stagnant"] + 1
    if stagnant >= threshold:
        return {"best": best_id, "stagnant": 0}, True
    return {"best": best_id, "stagnant": stagnant}, False
