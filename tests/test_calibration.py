"""Calibration comparison tests: predicted (B200 evaluator) vs measured pairs.

Offline coverage for the phase-4 report; also pins the new measured-bottleneck
field in run_eval pairs. No GPU, no API calls.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


def _pair(
    version="v1",
    backend="modal",
    hardware="NVIDIA B200",
    pred_us=1000.0,
    pred_bottleneck="TMA_MEMORY_BOUND",
    fitness=0.5,
    status="PASSED",
    latency_ms=2.0,
    meas_bottleneck="memory",
    note="predicted over 2 workload shapes",
    statuses=None,
    max_relative_error=0.001,
):
    return {
        "version": version,
        "backend": backend,
        "hardware": hardware,
        "note": note,
        "spec": {"tile_shape": [128, 256, 64], "format": "f16"},
        "predicted": {
            "latency_us": pred_us,
            "bottleneck": pred_bottleneck,
            "fitness": fitness,
        },
        "measured": {
            "eval_status": status,
            "passed": 2,
            "n_workloads": 2,
            "latency_ms": latency_ms,
            "bottleneck": meas_bottleneck,
            "max_relative_error": max_relative_error,
            **({"statuses": statuses} if statuses is not None else {}),
        },
    }


def _write_pairs(root: Path, run_id: str, pairs: list) -> None:
    folder = root / "runs" / run_id
    folder.mkdir(parents=True)
    with (folder / "calibration.jsonl").open("a") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair) + "\n")


class LoadPairsTest(unittest.TestCase):
    def test_collects_across_runs(self) -> None:
        from evaluators.b200.calibration import load_pairs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_pairs(root, "run-a", [_pair("v1"), _pair("v2")])
            _write_pairs(root, "run-b", [_pair("v1")])
            pairs = load_pairs(root)
        self.assertEqual(len(pairs), 3)
        self.assertEqual({p["run_id"] for p in pairs}, {"run-a", "run-b"})

    def test_missing_and_malformed(self) -> None:
        from evaluators.b200.calibration import load_pairs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(load_pairs(root), [])
            folder = root / "runs" / "run-a"
            folder.mkdir(parents=True)
            (folder / "calibration.jsonl").write_text(
                "not json\n" + json.dumps(_pair()) + "\n"
            )
            pairs = load_pairs(root)
        self.assertEqual(len(pairs), 1)


class AnalyzeTest(unittest.TestCase):
    def test_latency_math(self) -> None:
        from evaluators.b200.calibration import analyze

        pairs = [
            _pair("v1", pred_us=4000.0, latency_ms=1.0),
            _pair("v2", pred_us=1000.0, latency_ms=1.0),
            _pair("v3", pred_us=250.0, latency_ms=1.0),
        ]
        latency = analyze(pairs)["latency"]
        self.assertEqual(latency["n"], 3)
        self.assertAlmostEqual(latency["geomean_ratio"], 1.0)
        self.assertAlmostEqual(latency["median_ratio"], 1.0)
        self.assertAlmostEqual(latency["within_2x"], 1 / 3)

    def test_ranking(self) -> None:
        from evaluators.b200.calibration import analyze

        ordered = [
            _pair("v1", pred_us=500.0, latency_ms=0.5),
            _pair("v2", pred_us=1000.0, latency_ms=1.0),
            _pair("v3", pred_us=2000.0, latency_ms=2.0),
        ]
        self.assertAlmostEqual(analyze(ordered)["ranking"]["spearman"], 1.0)
        inverse = [
            _pair("v1", pred_us=2000.0, latency_ms=0.5),
            _pair("v2", pred_us=1000.0, latency_ms=1.0),
            _pair("v3", pred_us=500.0, latency_ms=2.0),
        ]
        self.assertAlmostEqual(analyze(inverse)["ranking"]["spearman"], -1.0)
        self.assertIsNone(analyze(ordered[:2])["ranking"]["spearman"])

    def test_failed_and_off_hardware_excluded(self) -> None:
        from evaluators.b200.calibration import analyze

        pairs = [
            _pair("v1"),
            _pair("v2", status="INCORRECT_NUMERICAL"),
            _pair("v3", hardware="NVIDIA GB10", backend="local"),
        ]
        analysis = analyze(pairs)
        self.assertEqual(analysis["latency"]["n"], 1)
        self.assertEqual(analysis["pairs_total"], 3)
        self.assertEqual(analyze(pairs, hardware=None)["latency"]["n"], 2)

    def test_clean_mixed_included_numerical_mixed_excluded(self) -> None:
        from evaluators.b200.calibration import analyze

        clean = _pair(
            "v1", status="MIXED",
            statuses={"PASSED": 15, "RUNTIME_ERROR": 2},
        )
        dirty = _pair(
            "v2", status="MIXED",
            statuses={"PASSED": 15, "INCORRECT_NUMERICAL": 2},
        )
        legacy = _pair("v3", status="MIXED")
        analysis = analyze([clean, dirty, legacy])
        self.assertEqual(analysis["latency"]["n"], 1)
        self.assertEqual(analysis["latency"]["mixed"], 1)

    def test_bottleneck_agreement(self) -> None:
        from evaluators.b200.calibration import analyze

        pairs = [
            _pair("v1", pred_bottleneck="TMA_MEMORY_BOUND", meas_bottleneck="memory"),
            _pair("v2", pred_bottleneck="COMPUTE_SATURATED", meas_bottleneck="compute"),
            _pair("v3", pred_bottleneck="COMPUTE_SATURATED", meas_bottleneck="memory"),
            _pair("v4", pred_bottleneck="SOMETHING_NEW", meas_bottleneck="memory"),
            _pair("v5", pred_bottleneck="TMA_MEMORY_BOUND", meas_bottleneck=None),
        ]
        bottleneck = analyze(pairs)["bottleneck"]
        self.assertEqual(bottleneck["n"], 3)
        self.assertAlmostEqual(bottleneck["agreement"], 2 / 3)
        self.assertEqual(bottleneck["confusion"]["memory>memory"], 1)
        self.assertEqual(bottleneck["confusion"]["compute>memory"], 1)

    def test_gates(self) -> None:
        from evaluators.b200.calibration import analyze

        pairs = [
            _pair("v1", pred_us=None, pred_bottleneck=None, fitness=None,
                  status="PASSED", note="gate: Shared memory capacity exceeded"),
            _pair("v2", pred_us=None, pred_bottleneck=None, fitness=None,
                  status="INCORRECT_NUMERICAL", note="gate: K tile 48 is odd"),
            _pair("v3"),
        ]
        gates = analyze(pairs)["gates"]
        self.assertEqual(gates["rejected"], 2)
        self.assertEqual(gates["false_rejects"], 1)
        self.assertEqual(gates["false_versions"], ["v1"])

    def test_coverage_notes(self) -> None:
        from evaluators.b200.calibration import analyze

        pairs = [
            _pair("v1"),
            _pair("v2", pred_us=None, pred_bottleneck=None, fitness=None,
                  note="tile config not found in source"),
        ]
        coverage = analyze(pairs)["coverage"]
        self.assertEqual(coverage, {"pairs": 2, "predicted": 1, "missing": 1})


class ReportTest(unittest.TestCase):
    def test_render_and_main(self) -> None:
        from evaluators.b200.calibration import analyze, main, render

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_pairs(root, "run-a", [_pair("v1"), _pair("v2")])
            text = render(analyze(load_pairs(root)))
        self.assertIn("2 pairs from 1 run", text)
        self.assertIn("Spearman", text)
        self.assertIn("bottleneck", text)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_pairs(root, "run-a", [_pair("v1")])
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(["--run", "run-a"], root=root)
        self.assertEqual(code, 0)
        self.assertIn("1 pair", out.getvalue())

    def test_main_json(self) -> None:
        from evaluators.b200.calibration import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_pairs(root, "run-a", [_pair("v1")])
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(["--json"], root=root)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())["pairs_total"], 1)

    def test_main_no_pairs(self) -> None:
        from evaluators.b200.calibration import main

        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with redirect_stdout(out):
                code = main([], root=Path(tmp))
        self.assertEqual(code, 0)
        self.assertIn("no calibration pairs", out.getvalue())


def load_pairs(root):
    from evaluators.b200.calibration import load_pairs as load

    return load(root)


if __name__ == "__main__":
    unittest.main()
