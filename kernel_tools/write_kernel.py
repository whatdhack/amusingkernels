"""write_kernel: save a kernel version for this run."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Context

EXTENSIONS = {
    "pytorch": ".py",
    "triton": ".py",
    "cute_dsl": ".py",
    "cutile": ".py",
    "cudnn_frontend": ".py",
    "cuda_cpp": ".cu",
    "cutlass": ".cpp",
    "cublas": ".cpp",
    "cudnn": ".cpp",
}

DEFINITION = {
    "type": "function",
    "name": "write_kernel",
    "description": "Write a kernel version: full source of one file implementing run() in "
    "destination-passing style (inputs first, pre-allocated outputs last, in-place, no return). "
    "Returns the version id (v1, v2, ...).",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "language": {
                "type": "string",
                "enum": sorted(EXTENSIONS),
                "description": "Solution language (Triton-first).",
            },
            "filename": {
                "type": "string",
                "description": "Source file name, e.g. kernel.py (default by language).",
            },
            "content": {"type": "string", "description": "Complete source code text."},
            "description": {
                "type": "string",
                "description": "What this version tries (one sentence).",
            },
        },
        "required": ["language", "content"],
    },
}


def run(ctx: Context, args: dict) -> dict:
    language = str(args.get("language", ""))
    if language not in EXTENSIONS:
        return {"error": f"unknown language: {language!r}"}
    content = str(args.get("content", ""))
    if not content.strip():
        return {"error": "content is empty"}
    filename = str(args.get("filename") or f"kernel{EXTENSIONS[language]}")
    if filename.startswith("/") or ".." in filename.split("/"):
        return {"error": f"filename must be relative with no '..': {filename!r}"}
    if not filename.endswith(EXTENSIONS[language]):
        return {
            "error": f"{language} sources must end in {EXTENSIONS[language]}: {filename!r}"
        }
    if "run" not in content:
        return {"error": "content must define a run() entry point"}
    version_id = f"v{len(ctx.record.data['versions']) + 1}"
    folder = ctx.root / "runs" / ctx.run_id / "kernels" / version_id
    folder.mkdir(parents=True)
    path = folder / filename
    path.write_text(content if content.endswith("\n") else content + "\n")
    version = ctx.record.add_version(
        source_path=str(path.relative_to(ctx.root)),
        language=language,
        entry_point=f"{filename}::run",
        description=str(args.get("description", "")),
    )
    return {
        "version": version["id"],
        "path": version["source_path"],
        "language": language,
        "entry_point": version["entry_point"],
    }
