"""KernelRecord: the versions-and-traces ledger for one run.

Lives at runs/<run-id>/record.json: every kernel version the agent wrote, each
eval it ran, and which version is currently best (correct + fastest).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class KernelRecord:
    def __init__(self, root: Path, run_id: str) -> None:
        self.path = Path(root) / "runs" / run_id / "record.json"
        try:
            self._data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self._data = {"versions": [], "best": None}

    @property
    def data(self) -> dict:
        return self._data

    def add_version(
        self,
        *,
        source_path: str,
        language: str,
        entry_point: str,
        description: str = "",
    ) -> dict:
        """Record a newly written kernel. Returns the version entry (id v1, v2, ...)."""
        version = {
            "id": f"v{len(self._data['versions']) + 1}",
            "source_path": source_path,
            "language": language,
            "entry_point": entry_point,
            "description": description,
            "solution_path": None,
            "evals": [],
        }
        self._data["versions"].append(version)
        self._save()
        return version

    def set_solution(self, version_id: str, solution_path: str) -> dict:
        """Attach a built solution.json to a version."""
        version = self.get_version(version_id)
        version["solution_path"] = solution_path
        self._save()
        return version

    def record_eval(self, version_id: str, result: dict[str, Any]) -> dict:
        """Append an eval result to a version and recompute the best version."""
        from .progress import now

        version = self.get_version(version_id)
        entry = {
            "time": now(),
            "eval_status": result.get("eval_status"),
            "passed": result.get("passed"),
            "n_workloads": result.get("n_workloads"),
            "latency_ms": result.get("latency_ms"),
            "sol_ms": result.get("sol_ms"),
            "pct_sol": result.get("pct_sol"),
            "reference_latency_ms": result.get("reference_latency_ms"),
            "speedup_factor": result.get("speedup_factor"),
            "max_relative_error": result.get("max_relative_error"),
            "trace": result.get("trace"),
        }
        version["evals"].append(entry)
        self._data["best"] = self._compute_best()
        self._save()
        return entry

    def get_version(self, version_id: str) -> dict:
        for version in self._data["versions"]:
            if version["id"] == version_id:
                return version
        raise KeyError(f"no such kernel version: {version_id}")

    def best_version(self) -> dict | None:
        """The best version (correct on all workloads + lowest latency), if any."""
        best = self._data.get("best")
        if best is None:
            return None
        return self.get_version(best)

    def _compute_best(self) -> str | None:
        best: str | None = None
        best_latency = float("inf")
        for version in self._data["versions"]:
            for eval in version["evals"]:
                if (
                    eval.get("eval_status") == "PASSED"
                    and eval.get("passed") == eval.get("n_workloads")
                    and isinstance(eval.get("latency_ms"), (int, float))
                    and eval["latency_ms"] < best_latency
                ):
                    best, best_latency = version["id"], eval["latency_ms"]
        return best

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=1))
        tmp.replace(self.path)
