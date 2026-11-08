"""Dual-model tests: critic client, version authorship, critic-on-failure.

No API calls, no GPU: the critic client is a stub with the same
chat.completions.create shape as the OpenAI SDK.
"""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from kernel_agent import critic
from kernel_agent.client import make_critic_client
from kernel_agent.record import KernelRecord
from kernel_agent.settings import critic_model
from kernel_tools import write_kernel

KERNEL = """import torch


def run(a, b, out):
    out[:] = a + b
"""

SUMMARY = {
    "eval_status": "INCORRECT_NUMERICAL",
    "passed": 0,
    "n_workloads": 2,
    "latency_ms": 1.5,
    "max_relative_error": 2.0,
    "backend": "local",
    "version": "v1",
}


def _stub_client(text="try smaller tiles", error=None):
    """A critic client stub capturing its create() kwargs."""
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        if error is not None:
            raise error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return client, captured


class DualModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.problem = self.root / "problem"
        self.problem.mkdir()
        (self.problem / "definition.json").write_text("{}")
        (self.problem / "workload.jsonl").write_text("{}\n")
        self.env = mock.patch.dict(
            os.environ,
            {"PROBLEM_DIR": str(self.problem), "AUTHOR": "test-agent"},
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

    def test_critic_model_default(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("GEMINI_API_MODEL", None)
            self.assertEqual(critic_model(), "gemini-4-argon")

    def test_critic_model_override(self) -> None:
        with mock.patch.dict(os.environ, {"GEMINI_API_MODEL": "other"}):
            self.assertEqual(critic_model(), "other")

    def test_make_critic_client_none_without_key(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("GEMINI_API_KEY", None)
            self.assertIsNone(make_critic_client())

    def test_make_critic_client_uses_env(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "key",
                "GEMINI_API_BASE_URL": "https://example.invalid/v1",
            },
        ):
            client = make_critic_client()
        self.assertIsNotNone(client)
        self.assertEqual(str(client.base_url), "https://example.invalid/v1/")

    def test_add_version_author(self) -> None:
        record = KernelRecord(self.root, "run-test")
        version = record.add_version(
            source_path="k",
            language="triton",
            entry_point="kernel.py::run",
            author="driver-model",
        )
        self.assertEqual(version["author"], "driver-model")
        self.assertEqual(record.get_version("v1")["author"], "driver-model")

    def test_add_version_author_defaults_none(self) -> None:
        record = KernelRecord(self.root, "run-test")
        version = record.add_version(
            source_path="k", language="triton", entry_point="kernel.py::run"
        )
        self.assertIsNone(version["author"])

    def test_write_kernel_stamps_author(self) -> None:
        with mock.patch.dict(os.environ, {"MODEL_API_MODEL": "driver-model"}):
            write_kernel.run(
                self.ctx, {"language": "triton", "content": KERNEL}
            )
        record = KernelRecord(self.root, "run-test")
        self.assertEqual(record.get_version("v1")["author"], "driver-model")

    def test_should_review_failed_eval(self) -> None:
        self.assertTrue(critic.should_review("run_eval", SUMMARY))
        self.assertTrue(
            critic.should_review("run_eval", {"eval_status": "MIXED"})
        )

    def test_should_review_ignores_passes_and_other_tools(self) -> None:
        self.assertFalse(
            critic.should_review("run_eval", {"eval_status": "PASSED"})
        )
        self.assertFalse(critic.should_review("run_eval", {"error": "nope"}))
        self.assertFalse(critic.should_review("write_kernel", SUMMARY))
        self.assertFalse(critic.should_review("run_eval", "not-a-dict"))

    def _record_with_source(self, source: str = KERNEL) -> KernelRecord:
        folder = self.root / "runs" / "run-test" / "kernels" / "v1"
        folder.mkdir(parents=True)
        (folder / "kernel.py").write_text(source)
        record = KernelRecord(self.root, "run-test")
        record.add_version(
            source_path="runs/run-test/kernels/v1/kernel.py",
            language="triton",
            entry_point="kernel.py::run",
            description="first try",
            author="driver-model",
        )
        return record

    def test_build_prompt_covers_source_and_summary(self) -> None:
        record = self._record_with_source()
        prompt = critic.build_prompt(
            root=self.root,
            run_id="run-test",
            record=record,
            version_id="v1",
            summary=dict(SUMMARY),
        )
        self.assertIn("out[:] = a + b", prompt)
        self.assertIn("INCORRECT_NUMERICAL", prompt)
        self.assertIn("0/2", prompt)
        self.assertIn("No passing version yet", prompt)

    def test_build_prompt_reports_best_and_trace_error(self) -> None:
        record = self._record_with_source()
        record.record_eval(
            "v1",
            {
                "eval_status": "PASSED",
                "passed": 2,
                "n_workloads": 2,
                "latency_ms": 0.5,
            },
        )
        trace = self.root / "runs" / "run-test" / "kernels" / "v1" / "trace.json"
        trace.write_text(
            '[{"evaluation": {"status": "INCORRECT_NUMERICAL", '
            '"error": "max error 2.0 at index 7"}}]'
        )
        prompt = critic.build_prompt(
            root=self.root,
            run_id="run-test",
            record=record,
            version_id="v1",
            summary={**SUMMARY, "trace": str(trace.relative_to(self.root))},
        )
        self.assertIn("v1 at 0.5", prompt)
        self.assertIn("max error 2.0 at index 7", prompt)

    def test_build_prompt_truncates_long_source(self) -> None:
        record = self._record_with_source("x = 1  # run\n" + "y = 2\n" * 2000)
        prompt = critic.build_prompt(
            root=self.root,
            run_id="run-test",
            record=record,
            version_id="v1",
            summary=dict(SUMMARY),
        )
        self.assertIn("(source truncated)", prompt)
        self.assertLess(len(prompt), critic.SOURCE_LIMIT + 3000)

    def test_review_returns_text(self) -> None:
        record = self._record_with_source()
        client, captured = _stub_client("  try smaller tiles  ")
        diagnosis = critic.review(
            client=client,
            model="critic-model",
            root=self.root,
            run_id="run-test",
            version_id="v1",
            summary=dict(SUMMARY),
            record=record,
        )
        self.assertEqual(diagnosis, "try smaller tiles")
        self.assertEqual(captured["model"], "critic-model")
        roles = [m["role"] for m in captured["messages"]]
        self.assertEqual(roles, ["system", "user"])

    def test_review_none_on_failure_or_empty(self) -> None:
        record = self._record_with_source()
        failing, _ = _stub_client(error=RuntimeError("boom"))
        self.assertIsNone(
            critic.review(
                client=failing,
                model="critic-model",
                root=self.root,
                run_id="run-test",
                version_id="v1",
                summary=dict(SUMMARY),
                record=record,
            )
        )
        empty, _ = _stub_client(text="   ")
        self.assertIsNone(
            critic.review(
                client=empty,
                model="critic-model",
                root=self.root,
                run_id="run-test",
                version_id="v1",
                summary=dict(SUMMARY),
                record=record,
            )
        )

    def test_review_failure_warns(self) -> None:
        record = self._record_with_source()
        failing, _ = _stub_client(error=RuntimeError("boom"))
        out = io.StringIO()
        with redirect_stdout(out):
            result = critic.review(
                client=failing,
                model="critic-model",
                root=self.root,
                run_id="run-test",
                version_id="v1",
                summary=dict(SUMMARY),
                record=record,
            )
        self.assertIsNone(result)
        self.assertIn("Critic review failed", out.getvalue())


class LoopCriticTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = mock.patch.dict(
            os.environ,
            {"PROBLEM_DIR": str(self.root), "GEMINI_API_MODEL": "critic-model"},
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.tmp.cleanup)

    def _driver(self, tool_name: str = "run_eval"):
        call = {
            "type": "function_call",
            "name": tool_name,
            "arguments": '{"version": "v1"}',
            "call_id": "call-1",
        }

        def create(**kwargs):
            return SimpleNamespace(
                output=[call], usage=SimpleNamespace(input_tokens=1, output_tokens=1)
            )

        return SimpleNamespace(responses=SimpleNamespace(create=create))

    def _setup_run(self):
        from kernel_agent.progress import new_run

        run_id = new_run(self.root, instructions="i", objective="o")
        folder = self.root / "runs" / run_id / "kernels" / "v1"
        folder.mkdir(parents=True)
        (folder / "kernel.py").write_text(KERNEL)
        KernelRecord(self.root, run_id).add_version(
            source_path=f"runs/{run_id}/kernels/v1/kernel.py",
            language="triton",
            entry_point="kernel.py::run",
            author="driver-model",
        )
        return run_id

    def test_loop_appends_critic_on_failed_eval(self) -> None:
        from kernel_agent.loop import _loop

        run_id = self._setup_run()
        client, captured = _stub_client("try smaller tiles")
        messages: list = [{"role": "user", "content": "o"}]
        with mock.patch(
            "kernel_tools.run", return_value=dict(SUMMARY)
        ):
            _loop(self.root, run_id, self._driver(), messages, "i", 1, critic=client)
        self.assertIn("model", captured)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("Second opinion", messages[-1]["content"])
        self.assertIn("try smaller tiles", messages[-1]["content"])

    def test_loop_skips_critic_on_pass(self) -> None:
        from kernel_agent.loop import _loop

        run_id = self._setup_run()
        client, captured = _stub_client("try smaller tiles")
        messages: list = [{"role": "user", "content": "o"}]
        passing = {**SUMMARY, "eval_status": "PASSED", "passed": 2}
        with mock.patch("kernel_tools.run", return_value=passing):
            _loop(self.root, run_id, self._driver(), messages, "i", 1, critic=client)
        self.assertNotIn("model", captured)
        self.assertFalse(
            any(
                m.get("role") == "user" and "Second opinion" in m.get("content", "")
                for m in messages
            )
        )

    def test_loop_without_critic_client(self) -> None:
        from kernel_agent.loop import _loop

        run_id = self._setup_run()
        messages: list = [{"role": "user", "content": "o"}]
        with mock.patch(
            "kernel_tools.run", return_value=dict(SUMMARY)
        ):
            result = _loop(
                self.root, run_id, self._driver(), messages, "i", 1, critic=None
            )
        self.assertEqual(result["status"], "paused")
        self.assertEqual(len(messages), 4)  # objective, last-turn, call, output


if __name__ == "__main__":
    unittest.main()
