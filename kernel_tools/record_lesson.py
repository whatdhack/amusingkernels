"""record_lesson: save a durable lesson for future runs (learnings.md)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Context

MAX_LESSON = 1000
MAX_PER_RUN = 8

DEFINITION = {
    "type": "function",
    "name": "record_lesson",
    "description": "Record one durable lesson for future runs (a failure cause with its fix, "
    "or a perf finding with numbers). Appended to learnings.md, which future runs read.",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "lesson": {
                "type": "string",
                "description": "The lesson in 1-3 sentences (under 1000 chars).",
            },
        },
        "required": ["lesson"],
    },
}


def run(ctx: Context, args: dict) -> dict:
    lesson = str(args.get("lesson", "")).strip()
    if not lesson:
        return {"error": "lesson is empty"}
    if len(lesson) > MAX_LESSON:
        return {"error": f"lesson is {len(lesson)} chars; keep it under {MAX_LESSON}"}
    path = ctx.root / "learnings.md"
    existing = path.read_text() if path.is_file() else ""
    if existing.count(ctx.run_id) >= MAX_PER_RUN:
        return {"error": f"already recorded {MAX_PER_RUN} lessons this run"}
    from kernel_agent.progress import now

    if not existing:
        existing = "# Kernel learnings\n\nDurable lessons across runs. Injected into each run's objective.\n"
    entry = f"\n## {now()[:10]} ({ctx.run_id})\n- {lesson}\n"
    path.write_text(existing + entry)
    lessons = sum(1 for line in path.read_text().splitlines() if line.startswith("## "))
    return {"recorded": True, "total_lessons": lessons}
