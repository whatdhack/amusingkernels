"""Knowledge layer: KernelWiki retrieval (kernelwiki.py) driven by measured
symptoms (symptoms.py). Ported from mini-kerngen's knowledge layer; the
symptom mapping here reads the run's traces instead of static analysis."""
from .kernelwiki import KernelWiki
from .symptoms import is_gemm_like, symptoms_for

__all__ = ["KernelWiki", "is_gemm_like", "symptoms_for"]
