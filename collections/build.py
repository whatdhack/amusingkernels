"""Assemble a collections submission from frozen solutions.

Reads solutions/<problem>/solution.json, validates each against the
submission schema (required fields, B200 target, DPS entry point, non-empty
sources), and writes collections/<name>/ with one file per definition plus a
manifest, then tars it. Run from the repo root:

    python collections/build.py [--name amusingkernels-fib-001]
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAME = "amusingkernels-l1-001"

COLLECTIONS = {
    "amusingkernels-l1-001": {
        "085_geglu_activation": "activation",
        "083_attention_score_value_matmul": "attention",
        "031_repeat_kv_attention_matmul": "attention",
        "077_whisper_decoder_output_projection": "gemm",
        "063_attention_output_reshape_and_projection": "gemm",
        "076_batched_expert_forward": "moe",
        "008_expert_output_weighted_index_add_accumulation": "moe",
    },
    "amusingkernels-fib-001": {
        "004_gemm_n128_k2048": "gemm",
        "005_gemm_n256_k7168": "gemm",
    },
}


class CollectionError(Exception):
    """The collection failed validation; carries the problem lines."""

    def __init__(self, problems: list[str]):
        super().__init__("\n".join(problems))
        self.problems = problems


def validate(solution: dict, path: Path) -> list[str]:
    problems = []
    for key in ("name", "definition", "author", "spec", "sources"):
        if key not in solution:
            problems.append(f"{path}: missing {key}")
    spec = solution.get("spec", {})
    for key in ("languages", "target_hardware", "entry_point"):
        if key not in spec:
            problems.append(f"{path}: spec missing {key}")
    if "B200" not in spec.get("target_hardware", []):
        problems.append(f"{path}: target_hardware lacks B200")
    if spec.get("destination_passing_style") is not True:
        problems.append(f"{path}: destination_passing_style is not true")
    sources = solution.get("sources", [])
    if not sources or not all(s.get("path") and s.get("content") for s in sources):
        problems.append(f"{path}: empty sources")
    entry = str(spec.get("entry_point", ""))
    if "::" not in entry or entry.split("::")[0] not in [s.get("path") for s in sources]:
        problems.append(f"{path}: entry_point file not in sources: {entry!r}")
    return problems


def build(root: Path, name: str) -> dict:
    """Validate and assemble one collection; returns the manifest summary.

    Raises KeyError for an unknown collection name, CollectionError when a
    frozen solution is missing or fails validation.
    """
    categories = COLLECTIONS[name]
    out = Path(root) / "collections" / name
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    problems: list[str] = []
    for problem in sorted(categories):
        src = Path(root) / "solutions" / problem / "solution.json"
        if not src.is_file():
            problems.append(f"missing frozen solution: {src}")
            continue
        solution = json.loads(src.read_text())
        problems.extend(validate(solution, src))
        dest = out / f"{solution['definition']}.json"
        dest.write_text(json.dumps(solution, indent=1) + "\n")
        entries.append(
            {
                "definition": solution["definition"],
                "category": categories[problem],
                "file": dest.name,
                "solution_name": solution["name"],
                "languages": solution["spec"]["languages"],
            }
        )
    if problems:
        raise CollectionError(problems)
    manifest = {
        "collection": name,
        "author": "amusingkernels",
        "count": len(entries),
        "solutions": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    tarball = Path(root) / "collections" / f"{name}.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(out, arcname=name)
    return {"entries": entries, "tarball": tarball}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=DEFAULT_NAME, choices=sorted(COLLECTIONS))
    args = parser.parse_args(argv)
    try:
        result = build(ROOT, args.name)
    except CollectionError as error:
        print("VALIDATION FAILED:")
        for problem in error.problems:
            print(" ", problem)
        return 1
    for entry in result["entries"]:
        print(f"  {entry['definition']} [{entry['category']}] <- {entry['file']}")
    tarball = result["tarball"]
    print(f"validated {len(result['entries'])} solutions")
    print(f"wrote {tarball} ({tarball.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
