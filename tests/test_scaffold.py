"""Offline scaffold tests: tools, record, solution builder (no API, no GPU)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import kernel_tools
from kernel_agent.loop import _objective
from kernel_agent.progress import _summary
from kernel_agent.record import KernelRecord
from kernel_agent.settings import solar_dir
from kernel_agent.sol import (
    PEAK_BW_BYTES_S,
    compute_peak_flops,
    ensure_macs,
    fused_bytes,
    resolve_dims,
    sol_ms,
    tensor_numel,
    trace_sols,
)
from kernel_tools import (
    build_solution,
    read_problem,
    record_lesson,
    run_eval,
    write_kernel,
)
from kernel_tools.run_eval import _summarize

DEFINITION = {
    "name": "rmsnorm_h4",
    "op_type": "rmsnorm",
    "axes": {
        "batch_size": {"type": "var"},
        "hidden_size": {"type": "const", "value": 4},
    },
    "inputs": {
        "input": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"},
        "weight": {"shape": ["hidden_size"], "dtype": "float32"},
        "eps": {"shape": None, "dtype": "float32"},
    },
    "outputs": {
        "output": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"},
    },
    "reference": "import torch\n\ndef run(input, weight, eps):\n    return input * weight",
}

WORKLOAD = {"uuid": "x", "axes": {"batch_size": 2}, "inputs": {}}

GOOD_KERNEL = """import torch
import triton
import triton.language as tl


def run(input, weight, eps, output):
    output[:] = input * weight
"""

BAD_ORDER_KERNEL = """import torch


def run(input, output, weight, eps):
    output[:] = input
"""

SYNTAX_KERNEL = "def run(:\n"


class ScaffoldTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.problem = self.root / "problem"
        self.problem.mkdir()
        (self.problem / "definition.json").write_text(json.dumps(DEFINITION))
        (self.problem / "workload.jsonl").write_text(json.dumps(WORKLOAD) + "\n")
        self.env = mock.patch.dict(
            os.environ,
            {"PROBLEM_DIR": str(self.problem), "AUTHOR": "test-agent"},
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.tmp.cleanup)
        self.ctx = SimpleNamespace(
            client=None, root=self.root, run_id="run-test",
            record=KernelRecord(self.root, "run-test"),
        )

    def test_definitions(self) -> None:
        definitions = kernel_tools.definitions()
        self.assertEqual(
            [d["name"] for d in definitions],
            ["read_problem", "write_kernel", "build_solution", "run_eval", "read_trace",
             "record_lesson"],
        )
        for definition in definitions:
            for key in ("type", "name", "description", "parameters"):
                self.assertIn(key, definition)

    def test_unknown_tool(self) -> None:
        result = kernel_tools.run(
            None, self.root, "run-test", {"name": "nope", "arguments": "{}"}
        )
        self.assertIn("unknown tool", result["error"])

    def test_bad_json_arguments(self) -> None:
        result = kernel_tools.run(
            None, self.root, "run-test", {"name": "read_problem", "arguments": "{oops"}
        )
        self.assertIn("not valid JSON", result["error"])

    def test_read_problem(self) -> None:
        result = read_problem.run(self.ctx, {})
        self.assertEqual(result["definition"], "rmsnorm_h4")
        self.assertEqual(
            result["dps_signature"], "run(input, weight, eps, output)"
        )
        self.assertEqual(result["n_workloads"], 1)
        self.assertIn("def run(input, weight, eps)", result["reference"])

    def test_write_and_build(self) -> None:
        written = write_kernel.run(
            self.ctx,
            {"language": "triton", "content": GOOD_KERNEL, "description": "v1"},
        )
        self.assertEqual(written["version"], "v1")
        self.assertEqual(written["entry_point"], "kernel.py::run")
        built = build_solution.run(self.ctx, {"version": "v1"})
        solution = json.loads((self.root / built["solution"]).read_text())
        self.assertEqual(solution["definition"], "rmsnorm_h4")
        self.assertEqual(solution["author"], "test-agent")
        self.assertEqual(solution["spec"]["languages"], ["triton"])
        self.assertIn("B200", solution["spec"]["target_hardware"])
        self.assertEqual(solution["spec"]["entry_point"], "kernel.py::run")
        self.assertTrue(solution["spec"]["destination_passing_style"])
        self.assertEqual(solution["sources"][0]["path"], "kernel.py")

    def test_build_rejects_bad_signature(self) -> None:
        write_kernel.run(self.ctx, {"language": "pytorch", "content": BAD_ORDER_KERNEL})
        result = build_solution.run(self.ctx, {"version": "v1"})
        self.assertIn("inputs then outputs", result["error"])

    def test_build_rejects_syntax_error(self) -> None:
        write_kernel.run(self.ctx, {"language": "pytorch", "content": SYNTAX_KERNEL})
        result = build_solution.run(self.ctx, {"version": "v1"})
        self.assertIn("syntax error", result["error"])

    def test_write_rejects_bad_filename(self) -> None:
        result = write_kernel.run(
            self.ctx,
            {"language": "triton", "filename": "../evil.py", "content": GOOD_KERNEL},
        )
        self.assertIn("relative", result["error"])
        result = write_kernel.run(
            self.ctx,
            {"language": "triton", "filename": "kernel.cu", "content": GOOD_KERNEL},
        )
        self.assertIn("must end in .py", result["error"])

    def test_run_eval_needs_solution(self) -> None:
        write_kernel.run(self.ctx, {"language": "triton", "content": GOOD_KERNEL})
        result = run_eval.run(self.ctx, {"version": "v1"})
        self.assertIn("build_solution first", result["error"])

    def test_run_eval_missing_binary(self) -> None:
        write_kernel.run(self.ctx, {"language": "triton", "content": GOOD_KERNEL})
        build_solution.run(self.ctx, {"version": "v1"})
        with mock.patch.dict(os.environ, {"SOL_EXECBENCH": "/nonexistent/sol-execbench"}):
            result = run_eval.run(self.ctx, {"version": "v1"})
        self.assertIn("evaluator not found", result["error"])

    def test_record_best(self) -> None:
        record = KernelRecord(self.root, "run-test")
        record.add_version(
            source_path="kernels/v1/kernel.py",
            language="triton",
            entry_point="kernel.py::run",
        )
        record.add_version(
            source_path="kernels/v2/kernel.py",
            language="triton",
            entry_point="kernel.py::run",
        )
        record.record_eval(
            "v1",
            {"eval_status": "PASSED", "passed": 2, "n_workloads": 2,
             "latency_ms": 0.5, "trace": "t1"},
        )
        record.record_eval(
            "v2",
            {"eval_status": "PASSED", "passed": 2, "n_workloads": 2,
             "latency_ms": 0.2, "trace": "t2"},
        )
        self.assertEqual(record.best_version()["id"], "v2")
        record.record_eval(
            "v1",
            {"eval_status": "PASSED", "passed": 2, "n_workloads": 2,
             "latency_ms": 0.1, "trace": "t3"},
        )
        self.assertEqual(KernelRecord(self.root, "run-test").best_version()["id"], "v1")

    def test_record_lesson(self) -> None:
        result = record_lesson.run(self.ctx, {"lesson": "Manual tanh beats the intrinsic."})
        self.assertTrue(result["recorded"])
        self.assertEqual(result["total_lessons"], 1)
        text = (self.root / "learnings.md").read_text()
        self.assertIn("Manual tanh beats the intrinsic.", text)
        self.assertIn("run-test", text)
        objective = _objective(self.root)
        self.assertIn("Prior learnings from earlier runs:", objective)
        self.assertIn("Manual tanh beats the intrinsic.", objective)

    def test_record_lesson_rejects_empty_and_long(self) -> None:
        self.assertIn("empty", record_lesson.run(self.ctx, {"lesson": "  "})["error"])
        self.assertIn(
            "under 1000",
            record_lesson.run(self.ctx, {"lesson": "x" * 1001})["error"],
        )

    def test_record_lesson_caps_per_run(self) -> None:
        for i in range(8):
            result = record_lesson.run(self.ctx, {"lesson": f"lesson {i}"})
            self.assertTrue(result.get("recorded"), result)
        self.assertIn("already recorded", record_lesson.run(
            self.ctx, {"lesson": "one too many"})["error"])

    def test_objective_without_learnings(self) -> None:
        self.assertNotIn("Prior learnings", _objective(self.root))

    def test_record_ignores_failed_for_best(self) -> None:
        record = KernelRecord(self.root, "run-test")
        record.add_version(
            source_path="k", language="triton", entry_point="kernel.py::run"
        )
        record.record_eval(
            "v1",
            {"eval_status": "INCORRECT_NUMERICAL", "passed": 0, "n_workloads": 1,
             "latency_ms": 0.01, "trace": "t"},
        )
        self.assertIsNone(record.best_version())


class SolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.problem = self.root / "problem"
        self.problem.mkdir()
        (self.problem / "definition.json").write_text(json.dumps(DEFINITION))
        (self.problem / "workload.jsonl").write_text(json.dumps(WORKLOAD) + "\n")
        self.env = mock.patch.dict(
            os.environ,
            {"PROBLEM_DIR": str(self.problem), "SOLAR_DIR": "/nonexistent-solar"},
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_resolve_dims(self) -> None:
        definition = {
            "axes": {
                "a": {"type": "var"},
                "c": {"type": "const", "value": 4},
                "e": {"type": "expr", "expression": "a * c + 1"},
            }
        }
        self.assertEqual(
            resolve_dims(definition, {"a": 3}), {"a": 3, "c": 4, "e": 13}
        )

    def test_tensor_numel(self) -> None:
        self.assertEqual(tensor_numel(["a", "c"], {"a": 3, "c": 4}), 12)
        self.assertEqual(tensor_numel(None, {}), 1)
        self.assertEqual(tensor_numel([], {}), 1)

    def test_fused_bytes(self) -> None:
        # input 2x4 f32 (32B) + weight 4 f32 (16B) + eps scalar (4B) + output (32B)
        self.assertEqual(fused_bytes(DEFINITION, {"batch_size": 2}), 84)

    def test_sol_ms_memory_bound(self) -> None:
        result = sol_ms(DEFINITION, WORKLOAD, macs=0)
        expected = 84 / PEAK_BW_BYTES_S * 1000
        self.assertAlmostEqual(result["sol_ms"], expected)
        self.assertEqual(result["bottleneck"], "memory")
        self.assertEqual(result["compute_ms"], 0.0)
        self.assertEqual(result["bytes"], 84)

    def test_fp32_peak_calibrated(self) -> None:
        # DEFINITION's first input is float32; calibrated against 076 (see sol.py).
        self.assertEqual(compute_peak_flops(DEFINITION), 24.0e12)

    def test_sol_ms_compute_bound(self) -> None:
        result = sol_ms(DEFINITION, WORKLOAD, macs=1e12)
        self.assertEqual(result["bottleneck"], "compute")
        self.assertAlmostEqual(
            result["sol_ms"], 1e12 * 2 / compute_peak_flops(DEFINITION) * 1000
        )

    def test_ensure_macs_falls_back_without_solar(self) -> None:
        macs = ensure_macs(self.root, "run-test", self.problem)
        self.assertEqual(macs, {"x": None})
        self.assertTrue((self.root / "runs" / "run-test" / "solar_macs.json").is_file())

    def test_trace_sols_matches_uuid(self) -> None:
        traces = [{"workload": {"uuid": "x"}, "evaluation": {"status": "PASSED"}}]
        sols = trace_sols(self.root, "run-test", traces)
        self.assertEqual(len(sols), 1)
        self.assertAlmostEqual(sols[0]["sol_ms"], 84 / PEAK_BW_BYTES_S * 1000)
        unknown = trace_sols(
            self.root, "run-test", [{"workload": {"uuid": "nope"}}]
        )
        self.assertEqual(unknown, [None])

    def test_summarize_reports_sol(self) -> None:
        traces = [
            {
                "evaluation": {
                    "status": "PASSED",
                    "correctness": {"max_relative_error": 0.0},
                    "performance": {"latency_ms": 1.0},
                }
            }
        ]
        summary = _summarize(traces, [{"sol_ms": 0.5}])
        self.assertAlmostEqual(summary["sol_ms"], 0.5)
        self.assertAlmostEqual(summary["pct_sol"], 50.0)

    def test_summary_line_shows_sol(self) -> None:
        line = _summary(
            json.dumps(
                {
                    "eval_status": "PASSED",
                    "latency_ms": 0.7114,
                    "sol_ms": 0.5989,
                    "pct_sol": 84.2,
                    "passed": 16,
                    "n_workloads": 16,
                }
            )
        )
        self.assertIn("0.711ms", line)
        self.assertIn("SOL 0.599ms, 84.2%", line)

    def test_sol_ms_b200_beats_gb10(self) -> None:
        gb10 = sol_ms(DEFINITION, WORKLOAD, macs=0)
        b200 = sol_ms(DEFINITION, WORKLOAD, macs=0, arch="B200")
        self.assertLess(b200["sol_ms"], gb10["sol_ms"])
        self.assertAlmostEqual(
            b200["sol_ms"] / gb10["sol_ms"], 273e9 / (5111.6 * 1.5e9)
        )

    def test_trace_sols_picks_arch_from_hardware(self) -> None:
        traces = [
            {
                "workload": {"uuid": "x"},
                "evaluation": {
                    "status": "PASSED",
                    "environment": {"hardware": "NVIDIA B200"},
                },
            },
            {
                "workload": {"uuid": "x"},
                "evaluation": {
                    "status": "PASSED",
                    "environment": {"hardware": "NVIDIA GB10"},
                },
            },
        ]
        sols = trace_sols(self.root, "run-test", traces)
        self.assertLess(sols[0]["sol_ms"], sols[1]["sol_ms"])

    def test_solar_dir(self) -> None:
        self.assertIsNone(solar_dir(self.root))
        checkout = self.root / "checkout"
        (checkout / "solar" / "perf").mkdir(parents=True)
        with mock.patch.dict(os.environ, {"SOLAR_DIR": str(checkout)}):
            self.assertEqual(solar_dir(self.root), checkout)


class ModalBackendTest(unittest.TestCase):
    TRACE = {
        "definition": "rmsnorm_h4",
        "workload": {"uuid": "x", "axes": {"batch_size": 2}, "inputs": {}},
        "evaluation": {
            "status": "PASSED",
            "correctness": {"max_relative_error": 0.001},
            "performance": {"latency_ms": 2.0},
            "environment": {"hardware": "NVIDIA B200", "libs": {}},
            "timestamp": "2026-01-01T00:00:00Z",
        },
    }

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.problem = self.root / "problem"
        self.problem.mkdir()
        (self.problem / "definition.json").write_text(json.dumps(DEFINITION))
        (self.problem / "workload.jsonl").write_text(json.dumps(WORKLOAD) + "\n")
        self.env = mock.patch.dict(
            os.environ,
            {
                "PROBLEM_DIR": str(self.problem),
                "EVAL_BACKEND": "modal",
                "SOLAR_DIR": "/nonexistent-solar",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.tmp.cleanup)
        self.ctx = SimpleNamespace(
            client=None,
            root=self.root,
            run_id="run-test",
            record=KernelRecord(self.root, "run-test"),
        )
        write_kernel.run(self.ctx, {"language": "triton", "content": GOOD_KERNEL})
        build_solution.run(self.ctx, {"version": "v1"})

    def test_modal_eval_parses_remote_trace(self) -> None:
        stdout = json.dumps(self.TRACE) + "\n"
        with (
            mock.patch("modal_eval.app"),
            mock.patch(
                "modal_eval.eval_kernel.remote",
                return_value={"returncode": 0, "stdout": stdout, "stderr": ""},
            ),
        ):
            result = run_eval.run(self.ctx, {"version": "v1"})
        self.assertEqual(result["eval_status"], "PASSED")
        self.assertEqual(result["backend"], "modal")
        self.assertAlmostEqual(result["latency_ms"], 2.0)
        self.assertIsNotNone(result["sol_ms"])
        self.assertTrue(
            (self.root / "runs" / "run-test" / "kernels" / "v1" / "trace.json").is_file()
        )

    def test_modal_eval_keeps_traces_despite_exit_code(self) -> None:
        stdout = json.dumps(self.TRACE) + "\n"
        with (
            mock.patch("modal_eval.app"),
            mock.patch(
                "modal_eval.eval_kernel.remote",
                return_value={"returncode": 1, "stdout": stdout, "stderr": ""},
            ),
        ):
            result = run_eval.run(self.ctx, {"version": "v1"})
        self.assertEqual(result["eval_status"], "PASSED")

    def test_modal_eval_failure(self) -> None:
        with (
            mock.patch("modal_eval.app"),
            mock.patch(
                "modal_eval.eval_kernel.remote",
                return_value={"returncode": 1, "stdout": "", "stderr": "boom"},
            ),
        ):
            result = run_eval.run(self.ctx, {"version": "v1"})
        self.assertIn("no JSON trace", result["error"])

    def test_modal_eval_timeout_pauses(self) -> None:
        with (
            mock.patch("modal_eval.app"),
            mock.patch(
                "modal_eval.eval_kernel.remote", return_value={"timeout": True}
            ),
        ):
            result = run_eval.run(self.ctx, {"version": "v1"})
        self.assertEqual(result["status"], "paused")

    def test_invalid_backend_rejected(self) -> None:
        with mock.patch.dict(os.environ, {"EVAL_BACKEND": "bogus"}):
            with self.assertRaises(RuntimeError):
                run_eval.run(self.ctx, {"version": "v1"})

    def test_summary_line_marks_modal(self) -> None:
        line = _summary(
            json.dumps(
                {
                    "eval_status": "PASSED",
                    "latency_ms": 1.0,
                    "passed": 1,
                    "n_workloads": 1,
                    "backend": "modal",
                }
            )
        )
        self.assertIn("[modal]", line)


if __name__ == "__main__":
    unittest.main()
