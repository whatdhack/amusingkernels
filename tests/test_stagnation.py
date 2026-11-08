"""Stagnation + KernelWiki tests: no API calls, no GPU, no real wiki checkout.

The wiki is faked per-test: tiny query.py/get_page.py scripts under a tmp dir,
exercising the real subprocess plumbing, page parsing, and tried-set rotation.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from kernel_agent import stagnation
from kernel_agent.record import KernelRecord
from kernel_agent.settings import kernelwiki_dir, stagnation_threshold
from kernel_tools.run_eval import _summarize
from knowledge import KernelWiki, is_gemm_like, symptoms_for
from knowledge.kernelwiki import NOT_ACTIONABLE, parse_page

KERNEL = """import torch


def run(a, b, out):
    out[:] = a + b
"""

FAILING = {
    "eval_status": "INCORRECT_NUMERICAL",
    "passed": 0,
    "n_workloads": 2,
    "latency_ms": 1.5,
    "version": "v1",
}

QUERY_PY = 'print("wiki/patterns/p.md")\n'

PATTERN_PAGE = """# wiki/patterns/p.md

---
id: pattern-p
title: "P"
type: pattern
candidate_techniques: [technique-pipeline-stages, technique-vectorized-loads]
---

Pattern idea first paragraph.

Pattern idea second paragraph.
"""

TECHNIQUE_PAGE = """# wiki/techniques/{tech}.md

---
id: {tech}
title: "{title}"
type: technique
architectures: [sm100]
confidence: source-reported
---

Technique idea first paragraph.

Technique idea second paragraph.
"""

GET_PAGE_PY = (
    "import sys\n"
    "PAGES = {\n"
    f'    "wiki/patterns/p.md": {PATTERN_PAGE!r},\n'
    '    "technique-pipeline-stages": '
    + repr(TECHNIQUE_PAGE.format(tech="technique-pipeline-stages", title="Stages"))
    + ",\n"
    '    "technique-vectorized-loads": '
    + repr(TECHNIQUE_PAGE.format(tech="technique-vectorized-loads", title="Loads"))
    + ",\n"
    "}\n"
    'print(PAGES.get(sys.argv[1], ""))\n'
)


def _fake_wiki(root: Path) -> Path:
    wiki = root / "wiki"
    scripts = wiki / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "query.py").write_text(QUERY_PY)
    (scripts / "get_page.py").write_text(GET_PAGE_PY)
    return wiki


def _stub_client(text="try a persistent kernel", error=None):
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


class StagnationUnitTest(unittest.TestCase):
    def test_first_best_sets_baseline(self) -> None:
        state, fired = stagnation.update(None, "v1", 3)
        self.assertEqual(state, {"best": "v1", "stagnant": 0})
        self.assertFalse(fired)

    def test_new_best_resets(self) -> None:
        state, fired = stagnation.update({"best": "v1", "stagnant": 2}, "v2", 3)
        self.assertEqual(state, {"best": "v2", "stagnant": 0})
        self.assertFalse(fired)

    def test_counts_and_fires_at_threshold(self) -> None:
        state, fired = stagnation.update({"best": "v1", "stagnant": 1}, "v1", 3)
        self.assertEqual(state, {"best": "v1", "stagnant": 2})
        self.assertFalse(fired)
        state, fired = stagnation.update(state, "v1", 3)
        self.assertEqual(state, {"best": "v1", "stagnant": 0})
        self.assertTrue(fired)

    def test_never_passing_counts_too(self) -> None:
        state: dict | None = None
        for _ in range(2):
            state, fired = stagnation.update(state, None, 3)
            self.assertFalse(fired)
        _, fired = stagnation.update(state, None, 3)
        self.assertTrue(fired)

    def test_zero_disables(self) -> None:
        state, fired = stagnation.update({"best": "v1", "stagnant": 9}, "v1", 0)
        self.assertFalse(fired)

    def test_is_eval_result_only_counts_run_eval(self) -> None:
        self.assertTrue(
            stagnation.is_eval_result("run_eval", {"eval_status": "MIXED"})
        )
        self.assertFalse(
            stagnation.is_eval_result("read_trace", {"eval_status": "MIXED"})
        )
        self.assertFalse(stagnation.is_eval_result("run_eval", {"error": "x"}))
        self.assertFalse(stagnation.is_eval_result("run_eval", "not-a-dict"))
        self.assertFalse(
            stagnation.is_eval_result(None, {"eval_status": "MIXED"})
        )

    def test_threshold_setting(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("STAGNATION_THRESHOLD", None)
            self.assertEqual(stagnation_threshold(), 3)
        with mock.patch.dict(os.environ, {"STAGNATION_THRESHOLD": "5"}):
            self.assertEqual(stagnation_threshold(), 5)
        with mock.patch.dict(os.environ, {"STAGNATION_THRESHOLD": "bogus"}):
            self.assertEqual(stagnation_threshold(), 3)

    def test_kernelwiki_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ):
                os.environ.pop("KERNELWIKI_DIR", None)
                repo = Path(tmp) / "repo"
                repo.mkdir()
                self.assertIsNone(kernelwiki_dir(repo))
                wiki = Path(tmp) / "KernelWiki"
                (wiki / "scripts").mkdir(parents=True)
                (wiki / "scripts" / "query.py").write_text("pass\n")
                self.assertEqual(kernelwiki_dir(repo), wiki)
                custom = _fake_wiki(repo)
                with mock.patch.dict(os.environ, {"KERNELWIKI_DIR": str(custom)}):
                    self.assertEqual(kernelwiki_dir(repo), custom)


class KnowledgeTest(unittest.TestCase):
    def test_is_gemm_like(self) -> None:
        self.assertTrue(is_gemm_like("083_attention_score_value_matmul", ""))
        self.assertTrue(is_gemm_like("x", "General matrix multiply (GEMM)."))
        self.assertTrue(is_gemm_like("004_gemm_n128_k2048", ""))
        self.assertFalse(is_gemm_like("085_geglu_activation", "GEGLU activation."))
        self.assertFalse(is_gemm_like("", ""))

    def test_symptoms_without_best_fall_back_to_kind(self) -> None:
        gemm = symptoms_for(best_eval=None, gemm_like=True)
        self.assertEqual(gemm[0][0], "compute-bound")
        mem = symptoms_for(best_eval=None, gemm_like=False)
        self.assertEqual(mem[0][0], "memory-bound")

    def test_symptoms_from_bottleneck(self) -> None:
        found = symptoms_for(
            best_eval={"latency_ms": 2.0, "pct_sol": 41.0, "bottleneck": "memory"},
            gemm_like=False,
        )
        self.assertEqual(found[0][0], "memory-bound")
        self.assertIn("41%", found[0][1])
        found = symptoms_for(
            best_eval={"latency_ms": 2.0, "pct_sol": 60.0, "bottleneck": "compute"},
            gemm_like=True,
        )
        self.assertEqual(found[0][0], "compute-bound")

    def test_symptoms_deduped(self) -> None:
        found = symptoms_for(
            best_eval={"latency_ms": 2.0, "pct_sol": 41.0, "bottleneck": "memory"},
            gemm_like=False,
        )
        self.assertEqual([s for s, _ in found], ["memory-bound"])

    def test_parse_page(self) -> None:
        page = parse_page(PATTERN_PAGE)
        self.assertIsNotNone(page)
        assert page is not None
        self.assertEqual(page.id, "pattern-p")
        self.assertEqual(
            page.candidate_techniques,
            ["technique-pipeline-stages", "technique-vectorized-loads"],
        )
        self.assertIn("first paragraph", page.summary)
        self.assertIsNone(parse_page("no frontmatter here"))

    def test_missing_wiki_yields_no_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wiki = KernelWiki(tmp)
            self.assertFalse(wiki.available)
            self.assertIsNone(wiki.prompt_block([("memory-bound", "x")], set()))

    def test_prompt_block_and_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wiki = KernelWiki(str(_fake_wiki(Path(tmp))))
            self.assertTrue(wiki.available)
            tried: set[str] = set()
            first = wiki.prompt_block([("memory-bound", "why")], tried)
            assert first is not None
            technique, block = first
            self.assertEqual(technique, "technique-pipeline-stages")
            self.assertIn("[KERNELWIKI technique-pipeline-stages", block)
            self.assertIn("pattern-p", block)
            self.assertIn("num_stages", block)
            self.assertIn("do not copy", block)
            tried.add(technique)
            second = wiki.prompt_block([("memory-bound", "why")], tried)
            assert second is not None
            self.assertEqual(second[0], "technique-vectorized-loads")
            self.assertNotIn(second[0], NOT_ACTIONABLE)


class ProposeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        folder = self.root / "runs" / "run-test" / "kernels" / "v1"
        folder.mkdir(parents=True)
        (folder / "kernel.py").write_text(KERNEL)
        self.record = KernelRecord(self.root, "run-test")
        self.record.add_version(
            source_path="runs/run-test/kernels/v1/kernel.py",
            language="triton",
            entry_point="kernel.py::run",
            description="first try",
            author="driver-model",
        )
        self.record.record_eval(
            "v1",
            {"eval_status": "INCORRECT_NUMERICAL", "passed": 0, "n_workloads": 2},
        )

    def test_recent_attempts(self) -> None:
        from kernel_agent.critic import recent_attempts

        lines = recent_attempts(self.record)
        self.assertEqual(len(lines), 1)
        self.assertIn("v1", lines[0])
        self.assertIn("INCORRECT_NUMERICAL", lines[0])

    def test_propose_returns_text(self) -> None:
        from kernel_agent.critic import propose

        client, captured = _stub_client("  try a persistent kernel  ")
        proposal = propose(
            client=client,
            model="critic-model",
            root=self.root,
            record=self.record,
            wiki_block="[KERNELWIKI technique-x]",
        )
        self.assertEqual(proposal, "try a persistent kernel")
        self.assertEqual(captured["model"], "critic-model")
        self.assertIn("technique-x", captured["messages"][1]["content"])
        self.assertIn("out[:] = a + b", captured["messages"][1]["content"])

    def test_propose_none_on_failure(self) -> None:
        from kernel_agent.critic import propose

        client, _ = _stub_client(error=RuntimeError("boom"))
        self.assertIsNone(
            propose(
                client=client,
                model="critic-model",
                root=self.root,
                record=self.record,
                wiki_block=None,
            )
        )

    def test_propose_failure_warns(self) -> None:
        import io
        from contextlib import redirect_stdout

        from kernel_agent.critic import propose

        client, _ = _stub_client(error=RuntimeError("boom"))
        out = io.StringIO()
        with redirect_stdout(out):
            result = propose(
                client=client,
                model="critic-model",
                root=self.root,
                record=self.record,
                wiki_block=None,
            )
        self.assertIsNone(result)
        self.assertIn("Critic proposal failed", out.getvalue())


class SummarizeBottleneckTest(unittest.TestCase):
    TRACE = {
        "evaluation": {
            "status": "PASSED",
            "correctness": {"max_relative_error": 0.0},
            "performance": {"latency_ms": 1.0},
        }
    }

    def test_majority_bottleneck(self) -> None:
        summary = _summarize(
            [self.TRACE, self.TRACE],
            [{"sol_ms": 0.5, "bottleneck": "memory"}, {"sol_ms": 0.5, "bottleneck": "memory"}],
        )
        self.assertEqual(summary["bottleneck"], "memory")

    def test_no_sols_no_bottleneck(self) -> None:
        summary = _summarize([self.TRACE], [{"sol_ms": 0.5}])
        self.assertIsNone(summary["bottleneck"])
        summary = _summarize([self.TRACE])
        self.assertIsNone(summary["bottleneck"])

    def test_record_eval_stores_bottleneck(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            record = KernelRecord(Path(tmp), "run-test")
            record.add_version(
                source_path="k", language="triton", entry_point="kernel.py::run"
            )
            entry = record.record_eval(
                "v1",
                {
                    "eval_status": "PASSED",
                    "passed": 1,
                    "n_workloads": 1,
                    "latency_ms": 0.5,
                    "bottleneck": "compute",
                },
            )
            self.assertEqual(entry["bottleneck"], "compute")


class LoopStagnationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.problem = self.root / "problem"
        self.problem.mkdir()
        (self.problem / "definition.json").write_text(
            '{"name": "005_gemm_n256_k7168",'
            ' "description": "General matrix multiply (GEMM)."}'
        )
        (self.problem / "workload.jsonl").write_text("{}\n")
        self.wiki = _fake_wiki(self.root)
        self.env = mock.patch.dict(
            os.environ,
            {
                "PROBLEM_DIR": str(self.problem),
                "KERNELWIKI_DIR": str(self.wiki),
                "GEMINI_API_MODEL": "critic-model",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.tmp.cleanup)

    def _driver(self):
        call = {
            "type": "function_call",
            "name": "run_eval",
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
            description="first try",
            author="driver-model",
        )
        return run_id

    def _stagnation_messages(self, messages):
        return [
            m
            for m in messages
            if m.get("role") == "user"
            and m.get("content", "").startswith("Stalled:")
        ]

    def test_fires_after_threshold_with_wiki_and_proposal(self) -> None:
        from kernel_agent.loop import _loop
        from kernel_agent.progress import load_progress

        run_id = self._setup_run()
        client, _ = _stub_client("try a persistent kernel")
        messages: list = [{"role": "user", "content": "o"}]
        with mock.patch("kernel_tools.run", return_value=dict(FAILING)):
            _loop(self.root, run_id, self._driver(), messages, "i", 3, critic=client)
        stalled = self._stagnation_messages(messages)
        self.assertEqual(len(stalled), 1)
        self.assertIn("technique-pipeline-stages", stalled[0]["content"])
        self.assertIn("try a persistent kernel", stalled[0]["content"])
        notes = load_progress(self.root, run_id)["notes"]
        self.assertEqual(notes["wiki_tried"], ["technique-pipeline-stages"])
        self.assertEqual(notes["stagnation"], {"best": None, "stagnant": 0})

    def test_second_window_rotates_technique(self) -> None:
        from kernel_agent.loop import _loop

        run_id = self._setup_run()
        client, _ = _stub_client("try a persistent kernel")
        messages: list = [{"role": "user", "content": "o"}]
        with mock.patch("kernel_tools.run", return_value=dict(FAILING)):
            _loop(self.root, run_id, self._driver(), messages, "i", 6, critic=client)
        stalled = self._stagnation_messages(messages)
        self.assertEqual(len(stalled), 2)
        self.assertIn("technique-pipeline-stages", stalled[0]["content"])
        self.assertIn("technique-vectorized-loads", stalled[1]["content"])

    def test_quiet_before_threshold(self) -> None:
        from kernel_agent.loop import _loop

        run_id = self._setup_run()
        client, _ = _stub_client("try a persistent kernel")
        messages: list = [{"role": "user", "content": "o"}]
        with mock.patch("kernel_tools.run", return_value=dict(FAILING)):
            _loop(self.root, run_id, self._driver(), messages, "i", 2, critic=client)
        self.assertEqual(self._stagnation_messages(messages), [])

    def test_wiki_only_without_critic(self) -> None:
        from kernel_agent.loop import _loop

        run_id = self._setup_run()
        messages: list = [{"role": "user", "content": "o"}]
        with mock.patch("kernel_tools.run", return_value=dict(FAILING)):
            _loop(self.root, run_id, self._driver(), messages, "i", 3, critic=None)
        stalled = self._stagnation_messages(messages)
        self.assertEqual(len(stalled), 1)
        self.assertIn("technique-pipeline-stages", stalled[0]["content"])
        self.assertNotIn("Fresh proposal", stalled[0]["content"])

    def test_nothing_to_say_appends_nothing(self) -> None:
        from kernel_agent.loop import _loop

        run_id = self._setup_run()
        messages: list = [{"role": "user", "content": "o"}]
        with (
            mock.patch.dict(os.environ, {"KERNELWIKI_DIR": "/nonexistent-wiki"}),
            mock.patch("kernel_tools.run", return_value=dict(FAILING)),
        ):
            _loop(self.root, run_id, self._driver(), messages, "i", 3, critic=None)
        self.assertEqual(self._stagnation_messages(messages), [])


if __name__ == "__main__":
    unittest.main()
