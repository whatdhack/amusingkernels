"""Collection builder tests: validation rules and pack assembly into tmp dirs."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _load_build():
    # `collections` is also a stdlib package, so import the builder by path.
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "collection_build", root / "collections" / "build.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_build = _load_build()
CollectionError = _build.CollectionError
build = _build.build
validate = _build.validate


def _solution(**overrides):
    solution = {
        "name": "004_gemm_n128_k2048_triton_v2",
        "definition": "004_gemm_n128_k2048",
        "author": "amusingkernels",
        "spec": {
            "languages": ["triton"],
            "target_hardware": ["B200", "LOCAL"],
            "entry_point": "kernel.py::run",
            "destination_passing_style": True,
        },
        "sources": [{"path": "kernel.py", "content": "def run(A, B, C): ..."}],
    }
    solution.update(overrides)
    return solution


class ValidateTest(unittest.TestCase):
    def test_good_solution_passes(self) -> None:
        self.assertEqual(validate(_solution(), Path("s.json")), [])

    def test_missing_fields_fail(self) -> None:
        bad = _solution()
        del bad["sources"]
        self.assertTrue(validate(bad, Path("s.json")))
        bad = _solution(
            spec={
                "languages": ["triton"],
                "target_hardware": ["LOCAL"],
                "entry_point": "kernel.py::run",
            }
        )
        problems = validate(bad, Path("s.json"))
        self.assertTrue(any("B200" in p for p in problems))
        self.assertTrue(any("destination_passing_style" in p for p in problems))

    def test_entry_point_must_match_sources(self) -> None:
        bad = _solution()
        bad["spec"]["entry_point"] = "other.py::run"
        self.assertTrue(
            any("entry_point" in p for p in validate(bad, Path("s.json")))
        )


class BuildTest(unittest.TestCase):
    def _root_with(self, solutions: dict) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for problem, solution in solutions.items():
            folder = root / "solutions" / problem
            folder.mkdir(parents=True)
            (folder / "solution.json").write_text(json.dumps(solution))
        return root

    def _collections(self, problems):
        return {
            "test-pack": {p: "gemm" for p in problems},
        }

    def test_builds_pack_manifest_and_tarball(self) -> None:
        root = self._root_with({"004_gemm_n128_k2048": _solution()})
        with mock.patch.object(
            _build, "COLLECTIONS",
            self._collections(["004_gemm_n128_k2048"]),
        ):
            result = build(root, "test-pack")
        out = root / "collections" / "test-pack"
        self.assertTrue((out / "004_gemm_n128_k2048.json").is_file())
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["count"], 1)
        self.assertEqual(manifest["solutions"][0]["category"], "gemm")
        self.assertTrue(result["tarball"].is_file())

    def test_missing_solution_fails(self) -> None:
        root = self._root_with({})
        with mock.patch.object(
            _build, "COLLECTIONS",
            self._collections(["004_gemm_n128_k2048"]),
        ):
            with self.assertRaises(CollectionError):
                build(root, "test-pack")

    def test_unknown_collection_fails(self) -> None:
        root = self._root_with({})
        with self.assertRaises(KeyError):
            build(root, "nope")


if __name__ == "__main__":
    unittest.main()
