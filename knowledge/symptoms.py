"""Map what the run measured to KernelWiki's symptom vocabulary. No LLM.

Sources, most specific first: the best version's roofline bottleneck
(memory/compute, recorded by run_eval), then the kind of problem when nothing
more specific is known. Every symptom comes with the measurement behind it,
which goes into the prompt as the reason the page was retrieved.
"""
from typing import Any, Dict, List, Optional, Tuple

# KernelWiki's symptom vocabulary (pattern pages' `symptoms:` fields); each resolves to a pattern page
VOCABULARY = ("compute-bound", "high-memory-throughput", "load-imbalance", "low-compute-utilization", "low-occupancy",
              "low-sm-utilization", "low-tensor-core-utilization", "memory-bound", "pipeline-stalls",
              "register-pressure", "register-spilling", "tail-effect", "wave-quantization")

GEMM_KEYWORDS = ("gemm", "matmul", "matrix multipl")


def is_gemm_like(name: str, description: str) -> bool:
    """True when the problem looks like a dense matrix multiply.

    definitions carry no op-type field, so this matches on the problem name
    and description text.
    """
    text = f"{name or ''} {description or ''}".lower()
    return any(keyword in text for keyword in GEMM_KEYWORDS)


def symptoms_for(
    *, best_eval: Optional[Dict[str, Any]], gemm_like: bool
) -> List[Tuple[str, str]]:
    """(symptom, why) pairs for the run, most specific first, without duplicates.

    `best_eval` is the best version's latest eval entry (latency_ms, pct_sol,
    bottleneck), or None when nothing has passed yet.
    """
    fallback = (
        ("compute-bound", "GEMM-like problem; no more specific symptom measured")
        if gemm_like
        else ("memory-bound", "bandwidth-bound problem; no more specific symptom measured")
    )
    if not best_eval:
        return [fallback]
    bottleneck = best_eval.get("bottleneck")
    if bottleneck == "compute":
        first: Tuple[str, str] = ("compute-bound", _why(best_eval))
    elif bottleneck == "memory":
        first = ("memory-bound", _why(best_eval))
    else:
        return [fallback]
    if first[0] == fallback[0]:
        return [first]
    return [first, fallback]


def _why(best_eval: Dict[str, Any]) -> str:
    latency = best_eval.get("latency_ms")
    latency_text = f"{latency:.3f}ms" if isinstance(latency, (int, float)) else "unknown latency"
    pct = best_eval.get("pct_sol")
    pct_text = f"{pct:.0f}% of roofline SOL" if isinstance(pct, (int, float)) else "SOL share unknown"
    return f"best kernel runs {latency_text} at {pct_text}; its passed workloads are {best_eval.get('bottleneck')}-bound"
