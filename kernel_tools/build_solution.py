"""build_solution: package a kernel version as a solution.json."""

from __future__ import annotations

import ast
import json
from typing import TYPE_CHECKING

from kernel_agent.settings import author, problem_dir

if TYPE_CHECKING:
    from . import Context

PYTHON_LANGUAGES = {"pytorch", "triton", "cute_dsl", "cutile", "cudnn_frontend"}

DEFAULT_DEPENDENCIES: dict[str, list[str]] = {
    "pytorch": ["torch"],
    "triton": ["torch", "triton >= 2.3"],
    "cute_dsl": ["torch", "cutlass"],
    "cutile": ["torch"],
    "cudnn_frontend": ["torch"],
    "cuda_cpp": [],
    "cutlass": [],
    "cublas": [],
    "cudnn": [],
}

DEFINITION = {
    "type": "function",
    "name": "build_solution",
    "description": "Build a solution.json from a recorded kernel version: validates the run() "
    "signature against the problem (inputs then outputs, destination-passing style) and writes "
    "the submission file next to the kernel.",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "version": {"type": "string", "description": "Kernel version id, e.g. v1."},
            "name": {
                "type": "string",
                "description": "Solution name (default <definition>_<language>_<version>).",
            },
            "target_hardware": {
                "type": "array",
                "items": {"type": "string"},
                "description": 'Target hardware (default ["B200", "LOCAL"]).',
            },
            "dependencies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Spec dependencies (default by language).",
            },
            "description": {"type": "string", "description": "Technique description."},
        },
        "required": ["version"],
    },
}


def run(ctx: Context, args: dict) -> dict:
    version_id = str(args.get("version", ""))
    try:
        version = ctx.record.get_version(version_id)
    except KeyError:
        return {"error": f"no such kernel version: {version_id}"}
    definition = json.loads(
        (problem_dir(ctx.root) / "definition.json").read_text()
    )
    source = ctx.root / version["source_path"]
    if not source.is_file():
        return {"error": f"kernel source is missing: {version['source_path']}"}
    content = source.read_text()
    expected = list(definition["inputs"]) + list(definition["outputs"])
    problem = _check_signature(content, version["language"], expected)
    if problem:
        return {"error": problem}
    solution = {
        "name": str(
            args.get("name")
            or f"{definition['name']}_{version['language']}_{version_id}"
        ),
        "definition": definition["name"],
        "author": author(),
        "spec": {
            "languages": [version["language"]],
            "target_hardware": args.get("target_hardware") or ["B200", "LOCAL"],
            "entry_point": version["entry_point"],
            "dependencies": args.get("dependencies")
            if args.get("dependencies") is not None
            else DEFAULT_DEPENDENCIES[version["language"]],
            "destination_passing_style": True,
        },
        "sources": [{"path": source.name, "content": content}],
    }
    if args.get("description"):
        solution["description"] = str(args["description"])
    solution_path = source.parent / "solution.json"
    solution_path.write_text(json.dumps(solution, indent=1) + "\n")
    ctx.record.set_solution(version_id, str(solution_path.relative_to(ctx.root)))
    return {
        "version": version_id,
        "solution": str(solution_path.relative_to(ctx.root)),
        "solution_name": solution["name"],
    }


def _check_signature(content: str, language: str, expected: list[str]) -> str | None:
    """Refuse kernels whose run() cannot match the definition's DPS signature."""
    if language in PYTHON_LANGUAGES:
        try:
            tree = ast.parse(content)
        except SyntaxError as error:
            return f"kernel.py has a syntax error: {error}"
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run":
                params = [a.arg for a in node.args.args + node.args.kwonlyargs]
                if params != expected:
                    return (
                        "run() parameters must be the definition inputs then outputs: "
                        f"expected ({', '.join(expected)}), got ({', '.join(params)})"
                    )
                return None
        return "kernel.py must define a run() function"
    if "run" not in content:
        return "kernel source must define a run() entry point"
    return None
