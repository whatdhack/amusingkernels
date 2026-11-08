"""Second opinion: a one-shot review from the critic model after a failed eval.

The driver model owns the tool loop; the critic never calls tools. After a
run_eval that did not fully pass, the loop asks the critic to diagnose the
failure from the kernel source plus the trace (plain chat completions, so any
OpenAI-compatible endpoint works) and appends the diagnosis as a user message
for the driver's next turn. A critic failure never stops the run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SOURCE_LIMIT = 6000
TRACE_LIMIT = 2000
MAX_TOKENS = 1000

SYSTEM = (
    "You review GPU kernels that just failed evaluation. Diagnose the root "
    "cause from the kernel source and the eval result: compile errors, then "
    "shape/dtype mismatches, then numerical errors, in that order. Cite the "
    "kernel lines involved. End with a concrete fix plan for the next version: "
    "what to change, not a full rewrite. Be specific and brief."
)

FRESH_SYSTEM = (
    "You unstick a GPU-kernel search that stopped improving. Given the best "
    "kernel so far, what was already tried, and an optional technique page, "
    "propose ONE different approach: a change of strategy, not a parameter "
    "tweak of the last version. Give concrete first steps. Be specific and "
    "brief."
)

ATTEMPT_LIMIT = 5


def should_review(tool_name: str, result: Any) -> bool:
    """True when a tool result deserves a critic review: a run_eval that ran
    but did not fully pass."""
    return (
        tool_name == "run_eval"
        and isinstance(result, dict)
        and result.get("eval_status") not in (None, "PASSED")
    )


def review(
    *,
    client: Any,
    model: str,
    root: Path,
    run_id: str,
    version_id: str,
    summary: dict,
    record: Any,
) -> str | None:
    """One critic diagnosis, or None when anything about it fails."""
    try:
        prompt = build_prompt(
            root=root,
            run_id=run_id,
            record=record,
            version_id=version_id,
            summary=summary,
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
        )
        text = (response.choices[0].message.content or "").strip()
        return text or None
    except Exception as error:
        print(f"Critic review failed ({type(error).__name__}): {error}", flush=True)
        return None


def propose(
    *,
    client: Any,
    model: str,
    root: Path,
    record: Any,
    wiki_block: str | None,
) -> str | None:
    """One fresh-eyes proposal for a stalled search, or None on any failure.

    Reads the best version (or the latest when nothing passed), what was
    already tried, and the retrieved technique block, if any.
    """
    try:
        target = record.best_version() or _latest(record)
        if target is None:
            return None
        lines = [
            f"Best so far: {target['id']} ({target.get('language')})"
            + (
                f": {target.get('description')}"
                if target.get("description")
                else ""
            ),
            "",
            _read_source(Path(root), target.get("source_path")),
            "",
            "Already tried:",
            *("- " + line for line in recent_attempts(record)),
        ]
        if wiki_block:
            lines += ["", wiki_block]
        lines += [
            "",
            "Propose ONE different optimization approach for the next version.",
        ]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FRESH_SYSTEM},
                {"role": "user", "content": "\n".join(lines)},
            ],
            max_tokens=MAX_TOKENS,
        )
        text = (response.choices[0].message.content or "").strip()
        return text or None
    except Exception as error:
        print(f"Critic proposal failed ({type(error).__name__}): {error}", flush=True)
        return None


def recent_attempts(record: Any, limit: int = ATTEMPT_LIMIT) -> list[str]:
    """One line per recent version: what it tried and how its eval went."""
    versions = record.data.get("versions", [])[-limit:]
    lines = []
    for version in versions:
        head = f"{version['id']} ({version.get('language')})"
        if version.get("description"):
            head += f": {version['description']}"
        evals = version.get("evals", [])
        if not evals:
            lines.append(f"{head} — not evaled")
            continue
        last = evals[-1]
        lines.append(
            f"{head} — {last.get('eval_status')} "
            f"{last.get('passed')}/{last.get('n_workloads')}"
        )
    return lines


def _latest(record: Any) -> dict | None:
    versions = record.data.get("versions", [])
    return versions[-1] if versions else None


def format_message(model: str, version_id: str, diagnosis: str) -> str:
    """The critic diagnosis as a user message for the driver's next turn."""
    return (
        f"Second opinion from {model} on {version_id} "
        "(its last eval did not fully pass). Consider this diagnosis before "
        "your next tool call; verify any claim against a fresh eval.\n\n"
        f"{diagnosis}"
    )


def build_prompt(
    *,
    root: Path,
    run_id: str,
    record: Any,
    version_id: str,
    summary: dict,
) -> str:
    """The review request: kernel source, eval summary, best so far, trace."""
    version = record.get_version(version_id)
    source = _read_source(Path(root), version.get("source_path"))
    lines = [
        f"Kernel {version_id} ({version.get('language')})"
        + (f": {version.get('description')}" if version.get("description") else ""),
        "",
        source,
        "",
        _summary_line(summary),
        _best_line(record),
        "",
        "First failing workload:",
        _failing_excerpt(Path(root), summary.get("trace")),
    ]
    return "\n".join(lines)


def _read_source(root: Path, source_path: Any) -> str:
    if not source_path:
        return "(kernel source unavailable)"
    try:
        text = (root / str(source_path)).read_text()
    except OSError:
        return "(kernel source unavailable)"
    if len(text) > SOURCE_LIMIT:
        return text[:SOURCE_LIMIT] + "\n...(source truncated)"
    return text


def _summary_line(summary: dict) -> str:
    parts = [
        f"Eval result: {summary.get('eval_status')}",
        f"{summary.get('passed')}/{summary.get('n_workloads')} workloads passed",
    ]
    if summary.get("latency_ms") is not None:
        parts.append(f"latency {summary['latency_ms']:.3f}ms")
    if summary.get("max_relative_error") is not None:
        parts.append(f"max relative error {summary['max_relative_error']}")
    if summary.get("backend"):
        parts.append(f"backend {summary['backend']}")
    return ", ".join(parts) + "."


def _best_line(record: Any) -> str:
    try:
        best = record.best_version()
    except KeyError:
        best = None
    if best is None:
        return "No passing version yet."
    latencies = [
        e.get("latency_ms")
        for e in best.get("evals", [])
        if e.get("eval_status") == "PASSED"
        and isinstance(e.get("latency_ms"), (int, float))
    ]
    if not latencies:
        return f"Current best: {best['id']} (passed, no latency recorded)."
    return f"Current best: {best['id']} at {min(latencies):.3f}ms."


def _failing_excerpt(root: Path, trace_path: Any) -> str:
    if not trace_path:
        return "(no trace recorded)"
    try:
        traces = json.loads((root / str(trace_path)).read_text())
    except (OSError, ValueError):
        return "(trace file unavailable)"
    if not isinstance(traces, list):
        traces = [traces]
    for trace in traces:
        evaluation = trace.get("evaluation") if isinstance(trace, dict) else None
        if isinstance(evaluation, dict) and evaluation.get("status") != "PASSED":
            return _clip(json.dumps(evaluation, indent=1))
    first = traces[0].get("evaluation", {}) if traces else {}
    return _clip(json.dumps(first, indent=1))


def _clip(text: str) -> str:
    if len(text) > TRACE_LIMIT:
        return text[:TRACE_LIMIT] + "\n...(trace truncated)"
    return text
