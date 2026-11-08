"""B200 evaluator tests: ported Petri-net/gate checks, the SOL adapter, and the
run_eval prediction wiring with pair-logging. No GPU, no API calls; the Modal
backend is stubbed and Tier-1 tests skip without tcgen05_model.
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from evaluators.b200 import B200Evaluator, hw
from evaluators.b200.petrinet_pipeline import PipelineConfig, simulate
from kernel_agent.progress import _summary
from kernel_agent.record import KernelRecord

EVALUATOR = B200Evaluator()

TRITON_GEMM = """import torch
import triton
import triton.language as tl


@triton.jit
def _matmul(a_ptr, b_ptr, c_ptr, M, N, K,
            stride_am, stride_ak, stride_bn, stride_bk,
            stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)


def run(A, B, C):
    M, K = A.shape
    N, _ = B.shape
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 256))
    _matmul[grid](A, B, C, M, N, K,
                  A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                  C.stride(0), C.stride(1),
                  BLOCK_M=128, BLOCK_N=256, BLOCK_K=64,
                  num_warps=8, num_stages=4)
"""

GEMM_DEFINITION = {
    "name": "005_gemm_n256_k7168",
    "description": "General matrix multiply (GEMM) C = A @ B.T.",
    "axes": {
        "M": {"type": "var"},
        "N": {"type": "const", "value": 256},
        "K": {"type": "const", "value": 7168},
    },
    "inputs": {
        "A": {"shape": ["M", "K"], "dtype": "float16"},
        "B": {"shape": ["N", "K"], "dtype": "float16"},
    },
    "outputs": {"C": {"shape": ["M", "N"], "dtype": "float16"}},
}

NORM_DEFINITION = {
    "name": "024_rmsnorm_h2048",
    "description": "Root Mean Square Normalization.",
    "axes": {
        "batch_size": {"type": "var"},
        "hidden_size": {"type": "const", "value": 2048},
    },
    "inputs": {
        "hidden_states": {
            "shape": ["batch_size", "hidden_size"],
            "dtype": "bfloat16",
        },
        "weight": {"shape": ["hidden_size"], "dtype": "bfloat16"},
    },
    "outputs": {"output": {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"}},
}

WORKLOADS = [
    {"uuid": "w1", "axes": {"M": 1}, "inputs": {}},
    {"uuid": "w2", "axes": {"M": 512}, "inputs": {}},
]


def _evaluate(**spec):
    base = {"format": "bf16", "tile_shape": (128, 128, 64), "pipeline_stages": 3}
    base.update(spec)
    return EVALUATOR.evaluate_candidate(base)


class PetriNetTest(unittest.TestCase):
    """Ported from mini-kerngen's Tier-2 test plan."""

    def test_per_sm_rates_reproduce_the_datasheet(self) -> None:
        for fmt, peak in (("bf16", 2250), ("e4m3", 4500), ("nvfp4", 9000), ("tf32", 1100)):
            with self.subTest(fmt=fmt):
                self.assertLess(abs(hw.peak_tflops(fmt) - peak) / peak, 0.1)

    def test_one_stage_on_memory_bound_shape_is_mostly_bubbles(self) -> None:
        result = simulate(
            PipelineConfig(128, 128, 64, stages=1, problem_m=128,
                           problem_n=16384, problem_k=4096)
        )
        self.assertGreater(result.bubble_ratio, 0.5)
        self.assertEqual(result.bottleneck, "TMA_MEMORY_BOUND")

    def test_more_stages_never_add_bubbles_and_saturate(self) -> None:
        ratios = [
            _evaluate(tile_shape=(128, 128, 64), pipeline_stages=s, ctas_per_sm=1)[
                "metrics"
            ]["bubble_ratio"]
            for s in range(2, 7)
        ]
        for earlier, later in zip(ratios, ratios[1:]):
            self.assertLessEqual(later, earlier)
        self.assertLess(ratios[-1], 0.15)

    def test_zero_bubble_invariant(self) -> None:
        result = _evaluate(tile_shape=(128, 256, 64), pipeline_stages=4)["metrics"]
        self.assertEqual(result["required_stages"], 4)
        self.assertEqual(result["bottleneck"], "COMPUTE_SATURATED")
        result = _evaluate(tile_shape=(128, 256, 64), pipeline_stages=3)["metrics"]
        self.assertEqual(result["bottleneck"], "TMA_MEMORY_BOUND")

    def test_tmem_column_limit(self) -> None:
        failed = _evaluate(tile_shape=(128, 600, 64), pipeline_stages=2)
        self.assertIn("TMEM column limit (512) exceeded", failed["error_reason"])
        failed = _evaluate(
            tile_shape=(128, 300, 64), pipeline_stages=2, acc_buffers=2
        )
        self.assertFalse(failed["passed_validation"])
        passed = _evaluate(
            tile_shape=(128, 256, 64), pipeline_stages=2, acc_buffers=2
        )
        self.assertTrue(passed["passed_validation"])

    def test_shared_memory_limit_is_227_kb_per_cta(self) -> None:
        failed = _evaluate(tile_shape=(256, 256, 64), pipeline_stages=4)
        self.assertIn("Shared memory capacity", failed["error_reason"])
        passed = _evaluate(tile_shape=(128, 256, 64), pipeline_stages=4)
        self.assertTrue(passed["passed_validation"])
        self.assertEqual(passed["metrics"]["smem_bytes_used"] // 1024, 192)

    def test_k_tile_multiple(self) -> None:
        failed = _evaluate(
            tile_shape=(128, 128, 48), pipeline_stages=4, format="e4m3"
        )
        self.assertIn("MMA K", failed["error_reason"])
        passed = _evaluate(
            tile_shape=(128, 128, 64), pipeline_stages=4, format="nvfp4"
        )
        self.assertTrue(passed["passed_validation"])

    def test_pipeline_improvements_ordered(self) -> None:
        steps = [
            dict(pipeline_stages=1, warp_specialized=False),
            dict(pipeline_stages=4, warp_specialized=False),
            dict(pipeline_stages=4, warp_specialized=True),
            dict(
                pipeline_stages=4,
                warp_specialized=True,
                persistent=True,
                acc_buffers=2,
            ),
        ]
        tflops = [
            _evaluate(tile_shape=(128, 256, 64), **s)["metrics"]["achieved_tflops"]
            for s in steps
        ]
        self.assertEqual(tflops, sorted(tflops))
        self.assertEqual(len(set(tflops)), 4)

    def test_result_shape(self) -> None:
        result = _evaluate()
        self.assertEqual(
            set(result), {"passed_validation", "error_reason", "fitness_score", "metrics"}
        )
        self.assertGreater(result["fitness_score"], 0.0)
        self.assertLessEqual(result["fitness_score"], 1.0)
        for key in (
            "achieved_tflops",
            "bubble_ratio",
            "bottleneck",
            "smem_bytes_used",
            "tmem_columns_used",
            "predicted_latency_us",
        ):
            self.assertIn(key, result["metrics"])


@unittest.skipUnless(
    importlib.util.find_spec("tcgen05_model") is not None,
    "tcgen05_model not installed (Tier 1 dormant)",
)
class Tier1Test(unittest.TestCase):
    def test_tf32_dot(self) -> None:
        from evaluators.b200 import functional_tcgen05 as tcgen05

        result = tcgen05.verify_dot([0x3F800000] * 8, [0x3F800000] * 8, "tf32")
        self.assertTrue(result.success)
        self.assertEqual(result.output_words, (0x41000000,))

    def test_bf16_dot(self) -> None:
        from evaluators.b200 import functional_tcgen05 as tcgen05

        result = tcgen05.verify_dot([1.0] * 8, [1.0] * 8, "bf16")
        self.assertTrue(result.success)
        self.assertEqual(result.value, 8.0)


class SolAdapterTest(unittest.TestCase):
    def test_extract_tiles_from_launch(self) -> None:
        from evaluators.b200.sol_adapter import extract_tiles

        tiles = extract_tiles(TRITON_GEMM)
        assert tiles is not None
        self.assertEqual(
            (tiles["tile_m"], tiles["tile_n"], tiles["tile_k"]), (128, 256, 64)
        )
        self.assertEqual(tiles["num_stages"], 4)
        self.assertFalse(tiles["tma"])
        self.assertFalse(tiles["persistent"])

    def test_extract_tiles_resolves_module_constants(self) -> None:
        from evaluators.b200.sol_adapter import extract_tiles

        code = TRITON_GEMM.replace("BLOCK_M=128", "BLOCK_M=BM").replace(
            "import triton.language as tl",
            "import triton.language as tl\n\nBM = 128",
        )
        tiles = extract_tiles(code)
        assert tiles is not None
        self.assertEqual(tiles["tile_m"], 128)

    def test_extract_tiles_detects_features(self) -> None:
        from evaluators.b200.sol_adapter import extract_tiles

        code = TRITON_GEMM.replace(
            "def run(A, B, C):",
            "def run(A, B, C):\n"
            "    desc = tl.make_tensor_descriptor(A, [128, 64], [512, 64])\n"
            "    NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count",
        )
        tiles = extract_tiles(code)
        assert tiles is not None
        self.assertTrue(tiles["tma"])
        self.assertTrue(tiles["persistent"])

    def test_extract_tiles_from_positional_launch(self) -> None:
        from evaluators.b200.sol_adapter import extract_tiles

        code = """import torch
import triton
import triton.language as tl


@triton.jit
def _k(a_ptr, M, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)


def run(A, B, C):
    M = A.shape[0]
    BM = 32
    _k[(4,)](A, M, BM, 128, 64, num_warps=4)
"""
        tiles = extract_tiles(code)
        assert tiles is not None
        self.assertEqual(
            (tiles["tile_m"], tiles["tile_n"], tiles["tile_k"]), (32, 128, 64)
        )

    def test_extract_tiles_rejects(self) -> None:
        from evaluators.b200.sol_adapter import extract_tiles

        self.assertIsNone(extract_tiles("def run(:\n"))
        self.assertIsNone(
            extract_tiles("import torch\n\n\ndef run(a, out):\n    out[:] = a\n")
        )
        self.assertIsNone(
            extract_tiles(
                TRITON_GEMM.replace("BLOCK_K=64", "BLOCK_K=BLOCK_UNKNOWN")
            )
        )

    def test_resolve_shapes(self) -> None:
        from evaluators.b200.sol_adapter import resolve_shapes

        self.assertEqual(
            resolve_shapes(GEMM_DEFINITION, WORKLOADS), [(1, 256, 7168), (512, 256, 7168)]
        )
        self.assertIsNone(resolve_shapes(NORM_DEFINITION, WORKLOADS))

    def test_map_format(self) -> None:
        from evaluators.b200.sol_adapter import map_format

        self.assertEqual(map_format("float16"), "f16")
        self.assertEqual(map_format("bfloat16"), "bf16")
        self.assertEqual(map_format("float8_e4m3fn"), "e4m3")
        self.assertEqual(map_format("float32"), "tf32")
        self.assertIsNone(map_format("float64"))

    def test_predict_version(self) -> None:
        from evaluators.b200.sol_adapter import predict_version

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            problem = root / "problem"
            problem.mkdir()
            (problem / "definition.json").write_text(json.dumps(GEMM_DEFINITION))
            (problem / "workload.jsonl").write_text(
                "".join(json.dumps(w) + "\n" for w in WORKLOADS)
            )
            source = root / "kernel.py"
            source.write_text(TRITON_GEMM)
            version = {
                "id": "v1",
                "language": "triton",
                "source_path": "kernel.py",
            }
            with mock.patch.dict(os.environ, {"PROBLEM_DIR": str(problem)}):
                prediction = predict_version(root=root, version=version)
        self.assertTrue(prediction["eligible"])
        self.assertGreater(prediction["predicted_latency_us"], 0)
        self.assertIn(
            prediction["predicted_bottleneck"],
            ("TMA_MEMORY_BOUND", "COMPUTE_SATURATED", "EPILOGUE_BOUND",
             "SMEM_CAPACITY_CONSTRAINED"),
        )
        self.assertEqual(prediction["spec"]["tile_shape"], [128, 256, 64])
        self.assertEqual(prediction["spec"]["format"], "f16")

    def test_predict_version_ineligible(self) -> None:
        from evaluators.b200.sol_adapter import predict_version

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            problem = root / "problem"
            problem.mkdir()
            (problem / "definition.json").write_text(json.dumps(NORM_DEFINITION))
            (problem / "workload.jsonl").write_text("{}\n")
            (root / "kernel.py").write_text(TRITON_GEMM)
            version = {"id": "v1", "language": "pytorch", "source_path": "kernel.py"}
            with mock.patch.dict(os.environ, {"PROBLEM_DIR": str(problem)}):
                prediction = predict_version(root=root, version=version)
        self.assertFalse(prediction["eligible"])
        self.assertIsNone(prediction["predicted_latency_us"])


class PredictionWiringTest(unittest.TestCase):
    TRACE = {
        "definition": "005_gemm_n256_k7168",
        "workload": {"uuid": "w1", "axes": {"M": 1}, "inputs": {}},
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
        (self.problem / "definition.json").write_text(json.dumps(GEMM_DEFINITION))
        (self.problem / "workload.jsonl").write_text(
            "".join(json.dumps(w) + "\n" for w in WORKLOADS)
        )
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

    def _version(self, language="triton", content=TRITON_GEMM):
        from kernel_tools import build_solution, write_kernel

        write_kernel.run(self.ctx, {"language": language, "content": content})
        build_solution.run(self.ctx, {"version": "v1"})

    def _modal(self, stdout):
        return (
            mock.patch("modal_eval.app"),
            mock.patch(
                "modal_eval.eval_kernel.remote",
                return_value={"returncode": 0, "stdout": stdout, "stderr": ""},
            ),
        )

    def test_modal_eval_attaches_prediction_and_logs_pair(self) -> None:
        from kernel_tools.run_eval import run as run_eval_run

        self._version()
        stdout = json.dumps(self.TRACE) + "\n"
        app_patch, remote_patch = self._modal(stdout)
        with app_patch, remote_patch:
            result = run_eval_run(self.ctx, {"version": "v1"})
        self.assertEqual(result["eval_status"], "PASSED")
        self.assertGreater(result["predicted_latency_us"], 0)
        self.assertIsNotNone(result["predicted_bottleneck"])
        pair_path = self.root / "runs" / "run-test" / "calibration.jsonl"
        self.assertTrue(pair_path.is_file())
        pair = json.loads(pair_path.read_text().splitlines()[-1])
        self.assertEqual(pair["version"], "v1")
        self.assertEqual(pair["backend"], "modal")
        self.assertEqual(pair["hardware"], "NVIDIA B200")
        self.assertAlmostEqual(pair["measured"]["latency_ms"], 2.0)
        self.assertEqual(pair["measured"]["bottleneck"], "memory")
        self.assertEqual(pair["measured"]["statuses"], {"PASSED": 1})
        self.assertAlmostEqual(pair["measured"]["max_relative_error"], 0.001)
        self.assertGreater(pair["predicted"]["latency_us"], 0)
        entry = KernelRecord(self.root, "run-test").get_version("v1")["evals"][-1]
        self.assertGreater(entry["predicted_latency_us"], 0)

    def test_non_gemm_eval_logs_no_pair(self) -> None:
        from kernel_tools.run_eval import run as run_eval_run

        (self.problem / "definition.json").write_text(json.dumps(NORM_DEFINITION))
        (self.problem / "workload.jsonl").write_text(
            json.dumps({"uuid": "w1", "axes": {"batch_size": 4}, "inputs": {}}) + "\n"
        )
        content = TRITON_GEMM.replace(
            "def run(A, B, C):", "def run(hidden_states, weight, output):"
        )
        self._version(language="triton", content=content)
        stdout = json.dumps(self.TRACE) + "\n"
        app_patch, remote_patch = self._modal(stdout)
        with app_patch, remote_patch:
            result = run_eval_run(self.ctx, {"version": "v1"})
        self.assertIsNone(result.get("predicted_latency_us"))
        self.assertFalse(
            (self.root / "runs" / "run-test" / "calibration.jsonl").is_file()
        )

    def test_summary_line_shows_prediction(self) -> None:
        line = _summary(
            json.dumps(
                {
                    "eval_status": "PASSED",
                    "latency_ms": 2.0,
                    "passed": 1,
                    "n_workloads": 1,
                    "predicted_latency_us": 1500.0,
                }
            )
        )
        self.assertIn("2.000ms", line)
        self.assertIn("[pred 1.500ms]", line)


if __name__ == "__main__":
    unittest.main()
