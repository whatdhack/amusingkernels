"""Calibration comparison: predicted (B200 evaluator) vs measured pairs.

Reads runs/*/calibration.jsonl (written by run_eval for eligible versions)
and reports latency agreement, ranking, and bottleneck agreement. Latency and
ranking use PASSED pairs plus MIXED pairs with no correctness failure (a
missing timing only shrinks the subset; a wrong result with a timing would
poison the mean). Only B200-measured pairs compare by default (--all lifts
that).

Usage: ./helper calibrate [--run RUN_ID] [--hardware SUBSTR | --all] [--json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

B200_HARDWARE = "B200"

# Evaluator bottleneck -> roofline bottleneck.
EVAL_TO_ROOFLINE = {
    "TMA_MEMORY_BOUND": "memory",
    "SMEM_CAPACITY_CONSTRAINED": "memory",
    "EPILOGUE_BOUND": "memory",
    "COMPUTE_SATURATED": "compute",
}


def load_pairs(root: Path) -> list[dict]:
    """Every calibration pair under runs/, each tagged with its run_id.

    Missing runs and malformed lines are skipped, never fatal.
    """
    pairs = []
    runs = Path(root) / "runs"
    if not runs.is_dir():
        return pairs
    for path in sorted(runs.glob("*/calibration.jsonl")):
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                pair = json.loads(line)
            except ValueError:
                continue
            if isinstance(pair, dict):
                pair["run_id"] = path.parent.name
                pairs.append(pair)
    return pairs


def analyze(
    pairs: list[dict], *, hardware: str | None = B200_HARDWARE, run_id: str | None = None
) -> dict:
    """The comparison. hardware=None compares every pair regardless of GPU."""
    scoped = [
        p
        for p in pairs
        if (run_id is None or p.get("run_id") == run_id)
    ]
    predicted = [p for p in scoped if _number(_predicted(p).get("latency_us"))]
    compared = [
        p
        for p in predicted
        if _comparable(_measured(p))
        and (hardware is None or hardware in str(p.get("hardware") or ""))
    ]
    ratios = [
        _predicted(p)["latency_us"] / (_measured(p)["latency_ms"] * 1000)
        for p in compared
    ]
    mixed = sum(1 for p in compared if _measured(p).get("eval_status") != "PASSED")
    return {
        "pairs_total": len(scoped),
        "runs": sorted({p.get("run_id", "") for p in scoped if p.get("run_id")}),
        "hardware_filter": hardware,
        "coverage": {
            "pairs": len(scoped),
            "predicted": len(predicted),
            "missing": len(scoped) - len(predicted),
        },
        "latency": {
            "n": len(ratios),
            "mixed": mixed,
            "geomean_ratio": _geomean(ratios),
            "median_ratio": _median(ratios),
            "within_2x": sum(1 for r in ratios if 0.5 <= r <= 2.0) / len(ratios)
            if ratios
            else None,
        },
        "ranking": {"n": len(compared), "spearman": _spearman(compared)},
        "bottleneck": _bottleneck(compared),
        "gates": _gates(scoped),
    }


def render(analysis: dict) -> str:
    """The comparison as human-readable text."""
    total = analysis["pairs_total"]
    if not total:
        return "calibration: no calibration pairs"
    runs = analysis["runs"]
    lines = [
        f"calibration: {total} pair{'s' if total != 1 else ''} "
        f"from {len(runs)} run{'s' if len(runs) != 1 else ''} "
        f"(hardware filter: {analysis['hardware_filter'] or 'all'})",
    ]
    coverage = analysis["coverage"]
    lines.append(
        f"coverage: {coverage['predicted']}/{coverage['pairs']} predicted, "
        f"{coverage['missing']} missing"
    )
    latency = analysis["latency"]
    if latency["n"]:
        mixed = f", {latency['mixed']} mixed" if latency.get("mixed") else ""
        lines.append(
            f"latency (PASSED + clean-mixed, n={latency['n']}{mixed}): "
            f"geomean pred/meas {_fmt(latency['geomean_ratio'])}, "
            f"median {_fmt(latency['median_ratio'])}, "
            f"within 2x {_fmt(latency['within_2x'])}"
        )
    else:
        lines.append("latency: no comparable pairs")
    ranking = analysis["ranking"]
    lines.append(
        f"ranking: Spearman {_fmt(ranking['spearman'])} (n={ranking['n']})"
    )
    bottleneck = analysis["bottleneck"]
    if bottleneck["n"]:
        confusion = ", ".join(
            f"{k}={v}" for k, v in sorted(bottleneck["confusion"].items())
        )
        lines.append(
            f"bottleneck: {_fmt(bottleneck['agreement'])} agreement "
            f"(n={bottleneck['n']}; {confusion})"
        )
    else:
        lines.append("bottleneck: no comparable pairs")
    gates = analysis["gates"]
    line = (
        f"gates: rejected {gates['rejected']}, "
        f"false rejects {gates['false_rejects']}"
    )
    if gates["false_versions"]:
        line += f" ({', '.join(gates['false_versions'])})"
    lines.append(line)
    return "\n".join(lines)


def main(argv: list[str] | None = None, root: Path | None = None) -> int:
    """./helper calibrate entry point. Prints the report; 0 always."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="", help="only this run id")
    parser.add_argument(
        "--hardware",
        default=B200_HARDWARE,
        help="only pairs measured on hardware containing this (default B200)",
    )
    parser.add_argument(
        "--all", action="store_true", help="compare every pair regardless of GPU"
    )
    parser.add_argument("--json", action="store_true", help="print the analysis as JSON")
    args = parser.parse_args(argv)
    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    pairs = load_pairs(base)
    analysis = analyze(
        pairs,
        hardware=None if args.all else args.hardware,
        run_id=args.run or None,
    )
    if args.json:
        print(json.dumps(analysis, indent=1))
    else:
        print(render(analysis))
    return 0


def _predicted(pair: dict) -> dict:
    predicted = pair.get("predicted")
    return predicted if isinstance(predicted, dict) else {}


def _measured(pair: dict) -> dict:
    measured = pair.get("measured")
    return measured if isinstance(measured, dict) else {}


def _comparable(measured: dict) -> bool:
    """True when the measured side can inform latency and ranking.

    PASSED pairs always count. MIXED pairs count only with a status
    breakdown showing no correctness failure (missing timings shrink the
    subset; wrong results with timings would poison the mean). Pairs from
    before the breakdown was recorded stay excluded.
    """
    if not _number(measured.get("latency_ms")):
        return False
    status = measured.get("eval_status")
    if status == "PASSED":
        return True
    if status != "MIXED":
        return False
    statuses = measured.get("statuses")
    if not isinstance(statuses, dict):
        return False
    return not any(str(s).startswith("INCORRECT") for s in statuses)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return float(value)


def _geomean(ratios: list[float]) -> float | None:
    if not ratios:
        return None
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _ranks(values: list[float]) -> list[float]:
    """Average ranks (1-based), ties sharing their mean rank."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def _spearman(compared: list[dict]) -> float | None:
    """Rank correlation of predicted vs measured latency, or None."""
    if len(compared) < 3:
        return None
    pred = [_predicted(p)["latency_us"] for p in compared]
    meas = [_measured(p)["latency_ms"] for p in compared]
    rank_pred, rank_meas = _ranks(pred), _ranks(meas)
    n = len(compared)
    mean_pred = sum(rank_pred) / n
    mean_meas = sum(rank_meas) / n
    cov = sum(
        (a - mean_pred) * (b - mean_meas) for a, b in zip(rank_pred, rank_meas)
    )
    var_pred = sum((a - mean_pred) ** 2 for a in rank_pred)
    var_meas = sum((b - mean_meas) ** 2 for b in rank_meas)
    if var_pred == 0 or var_meas == 0:
        return None
    return cov / math.sqrt(var_pred * var_meas)


def _bottleneck(compared: list[dict]) -> dict:
    confusion: dict[str, int] = {}
    for pair in compared:
        predicted = EVAL_TO_ROOFLINE.get(_predicted(pair).get("bottleneck"), "other")
        measured = _measured(pair).get("bottleneck")
        if predicted == "other" or measured not in ("memory", "compute"):
            continue
        key = f"{predicted}>{measured}"
        confusion[key] = confusion.get(key, 0) + 1
    total = sum(confusion.values())
    agree = sum(
        count for key, count in confusion.items() if key.split(">")[0] == key.split(">")[1]
    )
    return {
        "n": total,
        "agreement": agree / total if total else None,
        "confusion": confusion,
    }


def _gates(scoped: list[dict]) -> dict:
    rejected = [p for p in scoped if str(p.get("note", "")).startswith("gate:")]
    false = [p for p in rejected if _measured(p).get("eval_status") == "PASSED"]
    return {
        "rejected": len(rejected),
        "false_rejects": len(false),
        "false_versions": [str(p.get("version", "?")) for p in false],
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
