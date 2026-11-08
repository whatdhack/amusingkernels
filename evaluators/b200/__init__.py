"""B200 (sm_100) evaluator: GPU-less gates and predictions for Triton GEMMs.

  hw.py                  constants, each with its source
  functional_tcgen05.py  Tier 1: bit-exact tcgen05 arithmetic (tcgen05_model) for formats, scales, sparsity
  petrinet_pipeline.py   Tier 2: timed Petri net of the TMA -> tcgen05.mma -> epilogue pipeline
  b200_evaluator.py      the evaluate_candidate() API: hard gates, Tier 1, Tier 2, fitness
  sol_adapter.py         SOL-ExecBench kernel version -> evaluator spec (this repo's adapter)

hw, petrinet_pipeline, b200_evaluator and functional_tcgen05 are ported
verbatim from mini-kerngen's evaluators/b200; only sol_adapter is new. Tier 1
stays dormant: it needs sample input tiles the loop does not supply yet, and
its tcgen05_model dependency is optional (import-guarded in
functional_tcgen05).
"""
from .b200_evaluator import B200Evaluator

__all__ = ["B200Evaluator"]
