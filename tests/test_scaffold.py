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
from kernel_agent.record import KernelRecord
from kernel_tools import (
    build_solution,
    read_problem,
    record_lesson,
    run_eval,
    write_kernel,
)

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


if __name__ == "__main__":
    unittest.main()
