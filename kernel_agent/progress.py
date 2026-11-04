"""A run on disk: the console shows it turn by turn, and a stopped run resumes from it.

    run_id = new_run(root, instructions=..., objective=...)
    messages = [{"role": "user", "content": objective}]
    # each turn: response = client.responses.create(...); messages += response.output;
    #            result = tools.run(...) for the tool the model picked; append its function_call_output
    save_progress(root, run_id, messages, response)                     # end of every turn
    save_progress(root, run_id, messages, response, status="done")      # the turn that ends the run

    progress = load_progress(root, run_id)  # progress["messages"], ["status"], ["turn"]
    step = resume(root, run_id)             # continue a paused run: step["message"] for the model, or None

Everything goes in runs/<run-id>/progress.json.
"""

from __future__ import annotations

import datetime
import json
import uuid
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def new_run(root: Path, *, instructions: str, objective: str) -> str:
    """Start a run: a new folder under runs/. Returns its id."""
    from .settings import model

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"run-{stamp}-{uuid.uuid4().hex[:6]}"
    folder = _folder(root, run_id)
    folder.mkdir(parents=True)
    (folder / "instructions.md").write_text(instructions)
    (folder / "objective.md").write_text(objective)
    _write(
        root,
        run_id,
        {
            "run_id": run_id,
            "model": model(),
            "started": now(),
            "status": "running",
            "turn": 0,
            "messages": [],
            "turns": [],
            "notes": {},
        },
    )
    print(f"Run {run_id} started · model {model()}", flush=True)
    return run_id


def save_progress(
    root: Path,
    run_id: str,
    messages: list,
    response: Any = None,
    *,
    status: str = "running",
    **notes: Any,
) -> None:
    """End of a turn: save the conversation and the turn's tokens, print one line.

    `status` is "running" until the run stops (done, paused, ...).
    Without `response`, only the status changes (no model call this time).
    Anything in `notes` comes back from load_progress.
    """
    progress = load_progress(root, run_id)
    plain = [_plain(m) for m in messages]
    entry: dict[str, Any] = {"time": now(), "status": status, "end": len(plain)}
    if response is not None:
        output = [_plain(item) for item in response.output]
        before = progress["turns"][-1]["end"] if progress["turns"] else 0
        start = next(
            (i for i in range(before, len(plain)) if output and plain[i] == output[0]),
            before,
        )
        usage = response.usage
        progress["turn"] += 1
        entry.update(
            turn=progress["turn"],
            saw=start,
            output=start + len(output),
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )
    progress["turns"].append(entry)
    progress["notes"].update(notes)
    progress.update(messages=plain, status=status)
    _write(root, run_id, progress)
    print(_line(root, progress, entry), flush=True)


def load_progress(root: Path, run_id: str) -> dict:
    """A run's saved progress: messages, status, turn, notes."""
    path = _folder(root, run_id) / "progress.json"
    if not path.is_file():
        raise FileNotFoundError(f"no such run: {run_id}")
    return json.loads(path.read_text())


def resume(root: Path, run_id: str) -> dict:
    """Continue a paused run.

    Returns {"message": what to tell the model next, or None when there is nothing to
    continue, "status": the run's status, "note": why not}.
    """
    progress = load_progress(root, run_id)
    status = progress["status"]
    if status == "paused":
        return _go(
            "You have more turns. Continue optimizing, and finish by building a "
            "solution.json from your best correct kernel."
        )
    return _stop(status, f"run is {status}; nothing to resume")


# ---- helpers ------------------------------------------------------------------------


def _go(message: str) -> dict:
    return {"message": message, "status": "running", "note": ""}


def _stop(status: str, note: str) -> dict:
    return {"message": None, "status": status, "note": note}


def _folder(root: Path, run_id: str) -> Path:
    return Path(root) / "runs" / run_id


def _write(root: Path, run_id: str, progress: dict) -> None:
    path = _folder(root, run_id) / "progress.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, indent=1))
    tmp.replace(path)


def _plain(value: Any) -> Any:
    """SDK objects (such as the items of response.output) as plain JSON values."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _line(root: Path, progress: dict, entry: dict) -> str:
    """The terminal line for one saved turn."""
    run_id, status = progress["run_id"], entry["status"]
    if "turn" not in entry:
        return f"Run {run_id}: {status}"
    messages = progress["messages"]
    call = next(
        (
            m
            for m in messages[entry["saw"] : entry["output"]]
            if m.get("type") == "function_call"
        ),
        None,
    )
    if call is None:
        did = "no tool call (the model answered in text)"
    else:
        output = next(
            (
                m.get("output")
                for m in messages[entry["output"] : entry["end"]]
                if m.get("type") == "function_call_output"
                and m.get("call_id") == call.get("call_id")
            ),
            "",
        )
        did = f"{call.get('name')} → {_summary(output)}"
    line = f"Turn {entry['turn']}: {did} · versions {_versions(root, run_id)} · {status}"
    if status != "running":
        line += f"\nRun {run_id} stopped: {status} after {progress['turn']} turns"
    return line


def _summary(output: Any) -> str:
    try:
        result = json.loads(output) if isinstance(output, str) else output
    except ValueError:
        return str(output)[:80]
    if not isinstance(result, dict):
        return str(result)[:80]
    if "error" in result:
        return "refused: " + str(result["error"])[:80]
    if "eval_status" in result:
        text = str(result["eval_status"])
        if result.get("latency_ms") is not None:
            text += f" {result['latency_ms']:.3f}ms"
        if result.get("sol_ms") is not None:
            text += f" (SOL {result['sol_ms']:.3f}ms"
            if result.get("pct_sol") is not None:
                text += f", {result['pct_sol']:.1f}%"
            text += ")"
        if result.get("passed") is not None:
            text += f" ({result['passed']}/{result.get('n_workloads')} workloads)"
        return text
    if "solution" in result:
        return f"solution built ({result.get('version')})"
    if "version" in result:
        return f"kernel {result['version']} written"
    if "definition" in result:
        return f"problem {result['definition']}"
    return "ok"


def _versions(root: Path, run_id: str) -> int:
    record = Path(root) / "runs" / run_id / "record.json"
    try:
        return len(json.loads(record.read_text()).get("versions", []))
    except (OSError, ValueError):
        return 0
