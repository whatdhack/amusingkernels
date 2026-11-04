"""The kernel agent: the agentic loop.

run_agent works the problem; resume_run continues a run that stopped.
"""

from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI

import kernel_tools
from kernel_agent.progress import load_progress, new_run, resume, save_progress
from kernel_agent.settings import model, problem_dir

LAST_TURN = (
    "This is your last turn. Build a solution.json from your best correct kernel "
    "with build_solution using what you have."
)
NUDGE = "Call exactly one tool per turn. Continue with the next tool call."


def run_agent(*, root: Path, client: OpenAI, turns: int = 30) -> dict:
    root = Path(root)
    instructions = Path(__file__).with_name("system_instructions.md").read_text()
    objective = _objective(root)
    run_id = new_run(root, instructions=instructions, objective=objective)
    messages: list = [{"role": "user", "content": objective}]
    return _loop(root, run_id, client, messages, instructions, turns)


def resume_run(*, root: Path, run_id: str, client: OpenAI, turns: int = 15) -> dict:
    root = Path(root)
    progress = load_progress(root, run_id)
    step = resume(root, run_id)
    if step["message"] is None:
        fresh = load_progress(root, run_id)
        return {
            "run_id": run_id,
            "status": step["status"],
            "turn": fresh["turn"],
            "note": step["note"],
        }
    messages: list = list(progress["messages"])
    messages.append({"role": "user", "content": step["message"]})
    instructions = (root / "runs" / run_id / "instructions.md").read_text()
    return _loop(root, run_id, client, messages, instructions, turns)


def _loop(
    root: Path, run_id: str, client: OpenAI, messages: list, instructions: str, turns: int
) -> dict:
    definitions = kernel_tools.definitions()
    for i in range(turns):
        if i == turns - 1:
            messages.append({"role": "user", "content": LAST_TURN})
        try:
            response = client.responses.create(
                model=model(),
                instructions=instructions,
                input=messages,
                tools=definitions,
            )
        except Exception:
            save_progress(root, run_id, messages, status="paused")
            return {
                "run_id": run_id,
                "status": "paused",
                "turn": load_progress(root, run_id)["turn"],
            }
        made = [_as_dict(item) for item in response.output]
        messages.extend(made)
        call = next((m for m in made if _type(m) == "function_call"), None)
        if call is None:
            messages.append({"role": "user", "content": NUDGE})
            save_progress(root, run_id, messages, response, status="running")
            continue
        result = kernel_tools.run(client, root, run_id, call)
        call_id = call.get("call_id") if isinstance(call, dict) else None
        messages.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result),
            }
        )
        status = result.get("status", "running") if isinstance(result, dict) else "running"
        save_progress(root, run_id, messages, response, status=status)
        if status != "running":
            return {
                "run_id": run_id,
                "status": status,
                "turn": load_progress(root, run_id)["turn"],
            }
    save_progress(root, run_id, messages, status="paused")
    return {
        "run_id": run_id,
        "status": "paused",
        "turn": load_progress(root, run_id)["turn"],
    }


LEARNINGS_LIMIT = 3000


def _objective(root: Path) -> str:
    prob = problem_dir(root)
    definition = json.loads((prob / "definition.json").read_text())
    n_workloads = sum(1 for _ in (prob / "workload.jsonl").open())
    inputs = ", ".join(definition["inputs"])
    outputs = ", ".join(definition["outputs"])
    objective = (
        f"Optimize the SOL-ExecBench problem '{definition['name']}' "
        f"({prob}, {n_workloads} workload(s)).\n"
        f"Write a kernel `run({inputs}, {outputs})` that is numerically correct "
        f"first and then as fast as possible. Start by reading the problem."
    )
    learnings = _learnings(root)
    if learnings:
        objective += f"\n\nPrior learnings from earlier runs:\n{learnings}"
    return objective


def _learnings(root: Path) -> str:
    """Recent learnings.md entries for the objective (empty when none)."""
    path = Path(root) / "learnings.md"
    if not path.is_file():
        return ""
    text = path.read_text()
    if len(text) <= LEARNINGS_LIMIT:
        return text
    return "...(earlier learnings trimmed)\n" + text[-LEARNINGS_LIMIT:]


def _as_dict(item):
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json", exclude_none=True)
    return item


def _type(item) -> str | None:
    return item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
