"""SOL-ExecBench versions -> B200 evaluator specs, plus calibration pair logging.

For a Triton kernel on a GEMM-like problem, tile sizes come from the host
wrapper's launch site (adapting mini-kerngen triton_check.py's literal
resolution, without compiling anything), problem shapes from the definition
and workloads, and the operand format from the input dtypes. Shared memory,
TMEM use and CTA occupancy are left for the evaluator to estimate, and Tier 1
stays dormant (no sample input tiles are supplied).

predict_version() never raises: anything it cannot extract or evaluate comes
back as a note, and eligible attempts are logged as calibration pairs for the
phase-4 comparison of predicted vs measured B200 latency.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from knowledge import is_gemm_like
from kernel_agent.settings import problem_dir
from kernel_agent.sol import resolve_dims

from .b200_evaluator import B200Evaluator

TILE_ALIASES = {
    "tile_m": ("BLOCK_M", "BM", "TILE_M"),
    "tile_n": ("BLOCK_N", "BN", "TILE_N"),
    "tile_k": ("BLOCK_K", "BK", "TILE_K"),
}
FORMAT_BY_DTYPE = {
    "float16": "f16",
    "half": "f16",
    "bfloat16": "bf16",
    "float8_e4m3fn": "e4m3",
    "float8_e4m3fnuz": "e4m3",
    "float8_e5m2": "e5m2",
    # fp32 GEMM-like Triton kernels use tl.dot, which runs on tensor cores.
    "float32": "tf32",
}
FP8_RE = re.compile(r"float8e4|float8_e4m3|fp8e4|e4m3")
FP8_E5_RE = re.compile(r"float8e5|float8_e5m2|fp8e5|e5m2")
PERSISTENT_RE = re.compile(r"NUM_SMS|num_sms|multi_processor_count|NUM_PROGRAMS")
DEFAULT_NUM_STAGES = 3

_EVALUATOR = B200Evaluator()


def predict_version(*, root: Path, version: dict) -> dict:
    """B200 prediction for one kernel version. Never raises.

    Returns {"eligible", "note", "spec", "predicted_latency_us",
    "predicted_bottleneck", "predicted_fitness"}. Eligible means the version
    is a Triton kernel on a GEMM-like problem and deserves a calibration
    pair; anything further that fails leaves the prediction empty with a
    note saying why.
    """
    blank: dict = {
        "eligible": False,
        "note": "",
        "spec": None,
        "predicted_latency_us": None,
        "predicted_bottleneck": None,
        "predicted_fitness": None,
    }
    try:
        return _predict(Path(root), version, blank)
    except Exception as error:
        blank["note"] = f"prediction failed: {error}"
        return blank


def log_pair(root: Path, run_id: str, pair: dict) -> None:
    """Append one calibration pair to runs/<run-id>/calibration.jsonl."""
    try:
        path = Path(root) / "runs" / run_id / "calibration.jsonl"
        with path.open("a") as handle:
            handle.write(json.dumps(pair) + "\n")
    except OSError:
        pass


def extract_tiles(code: str) -> Optional[Dict[str, Any]]:
    """Tile config + features from Triton source, or None when unknown.

    BLOCK sizes and num_stages come from the first kernel launch site (names
    bound to literals resolve; an @triton.autotune config is the fallback).
    TMA, warp specialization, persistent grids and narrowed operands are read
    from the source text. Strict: a missing tile size means no prediction
    rather than a guessed one.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    jit = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and _is_jit(node)
    }
    if not jit:
        return None
    sites = _launch_sites(tree)
    match = next(((n, s) for n, s in sites.items() if n in jit), None)
    if match is None and sites:
        match = next(iter(sites.items()))
    if match is None:
        return None
    target, site = match
    params = _jit_params(tree, target)
    positionals = site.get("positional", [])
    auto_const, auto_opts = _first_autotune_config(tree)
    tiles: Dict[str, Any] = {}
    for key, names in TILE_ALIASES.items():
        value = None
        for name in names:
            candidate = site["constexprs"].get(name, auto_const.get(name))
            if _positive_int(candidate):
                value = candidate
                break
            if name in params:
                slot = params.index(name)
                if slot < len(positionals) and _positive_int(positionals[slot]):
                    value = positionals[slot]
                    break
        if value is None:
            return None
        tiles[key] = value
    stages = site["options"].get("num_stages", auto_opts.get("num_stages", DEFAULT_NUM_STAGES))
    if not isinstance(stages, int) or stages <= 0:
        stages = DEFAULT_NUM_STAGES
    tiles["num_stages"] = stages
    tiles["tma"] = "make_tensor_descriptor" in code
    tiles["warp_specialized"] = "warp_specialize" in code
    tiles["persistent"] = bool(PERSISTENT_RE.search(code))
    tiles["fp8"] = (
        "e5m2" if FP8_E5_RE.search(code) else ("e4m3" if FP8_RE.search(code) else None)
    )
    return tiles


def resolve_shapes(definition: dict, workloads: list) -> Optional[List[Tuple[int, int, int]]]:
    """(M, N, K) per workload from the two matrix inputs, or None.

    The K axis is the dimension name shared by both 2-D input shapes (which
    covers both B[N, K] and B[K, N] layouts); M and N are each shape's other
    dimension. Workloads whose dims do not resolve to ints are skipped.
    """
    inputs = definition.get("inputs", {})
    mats = [
        (name, info["shape"])
        for name, info in inputs.items()
        if isinstance(info.get("shape"), list) and len(info["shape"]) == 2
    ]
    if len(mats) < 2:
        return None
    first, second = mats[0][1], mats[1][1]
    shared = [d for d in first if isinstance(d, str) and d in second]
    if len(shared) != 1:
        return None
    k_name = shared[0]
    m_names = [d for d in first if d != k_name]
    n_names = [d for d in second if d != k_name]
    if len(m_names) != 1 or len(n_names) != 1:
        return None
    shapes = []
    for workload in workloads or []:
        dims = resolve_dims(definition, (workload or {}).get("axes", {}) or {})
        values = [_dim_value(m_names[0], dims), _dim_value(n_names[0], dims), _dim_value(k_name, dims)]
        if all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in values):
            shapes.append((values[0], values[1], values[2]))
    return shapes or None


def map_format(dtype: Any) -> Optional[str]:
    """Evaluator format for a definition dtype, or None when unsupported."""
    return FORMAT_BY_DTYPE.get(str(dtype)) if dtype is not None else None


def _predict(root: Path, version: dict, blank: dict) -> dict:
    if version.get("language") != "triton":
        blank["note"] = f"not a triton kernel ({version.get('language')})"
        return blank
    prob = problem_dir(root)
    definition = json.loads((prob / "definition.json").read_text())
    if not is_gemm_like(
        str(definition.get("name", "")), str(definition.get("description", ""))
    ):
        blank["note"] = "not a GEMM-like problem"
        return blank
    blank["eligible"] = True
    try:
        source = (root / str(version.get("source_path", ""))).read_text()
    except OSError:
        blank["note"] = "kernel source unreadable"
        return blank
    tiles = extract_tiles(source)
    if tiles is None:
        blank["note"] = "tile config not found in source"
        return blank
    workloads = [
        json.loads(line)
        for line in (prob / "workload.jsonl").read_text().splitlines()
        if line.strip()
    ]
    shapes = resolve_shapes(definition, workloads)
    if not shapes:
        blank["note"] = "no resolvable (M, N, K) workload shapes"
        return blank
    operand = _operand_format(definition, tiles["fp8"])
    if operand is None:
        blank["note"] = "unsupported or mixed matrix dtypes"
        return blank
    spec = {
        "tile_shape": [tiles["tile_m"], tiles["tile_n"], tiles["tile_k"]],
        "pipeline_stages": tiles["num_stages"],
        "format": operand,
        "persistent": tiles["persistent"],
        "tma": tiles["tma"],
        "warp_specialized": tiles["warp_specialized"],
    }
    blank["spec"] = spec
    latencies, bottlenecks, fitnesses = [], [], []
    for shape in shapes:
        result = _EVALUATOR.evaluate_candidate(
            {
                "tile_shape": tuple(spec["tile_shape"]),
                "pipeline_stages": spec["pipeline_stages"],
                "format": operand,
                "is_sparse": False,
                "problem_shape": shape,
                "persistent": spec["persistent"],
                "tma": spec["tma"],
                "warp_specialized": spec["warp_specialized"],
                "acc_buffers": 1,
            }
        )
        if not result["passed_validation"]:
            blank["note"] = f"gate: {result['error_reason']}"
            return blank
        metrics = result["metrics"]
        latencies.append(metrics["predicted_latency_us"])
        bottlenecks.append(metrics["bottleneck"])
        fitnesses.append(result["fitness_score"])
    blank["predicted_latency_us"] = sum(latencies) / len(latencies)
    blank["predicted_bottleneck"] = Counter(bottlenecks).most_common(1)[0][0]
    blank["predicted_fitness"] = sum(fitnesses) / len(fitnesses)
    blank["note"] = f"predicted over {len(shapes)} workload shapes"
    return blank


def _operand_format(definition: dict, fp8_override: Optional[str]) -> Optional[str]:
    if fp8_override:
        return fp8_override
    inputs = definition.get("inputs", {})
    mats = [
        info
        for info in inputs.values()
        if isinstance(info.get("shape"), list) and len(info["shape"]) == 2
    ]
    formats = {map_format(info.get("dtype")) for info in mats[:2]}
    if len(formats) != 1:
        return None
    return next(iter(formats))


def _dim_value(entry: Any, dims: dict) -> Any:
    if isinstance(entry, int) and not isinstance(entry, bool):
        return entry
    if isinstance(entry, str):
        return dims.get(entry)
    return None


def _is_jit(node: ast.AST) -> bool:
    return isinstance(node, ast.FunctionDef) and any("triton.jit" in ast.unparse(d) for d in node.decorator_list)


def _is_literal(node: ast.AST) -> bool:
    """A literal, or tl.constexpr(<literal>)."""
    try:
        ast.literal_eval(node)
        return True
    except ValueError:
        pass
    return (isinstance(node, ast.Call) and ast.unparse(node.func) in ("tl.constexpr", "triton.language.constexpr")
            and len(node.args) == 1 and not node.keywords and _is_literal(node.args[0]))


def _literal_value(node: ast.AST) -> Tuple[bool, Any]:
    """(True, value) for a literal or tl.constexpr(<literal>), else (False, None)."""
    if isinstance(node, ast.Call) and _is_literal(node):
        node = node.args[0]
    try:
        return True, ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return False, None


_UNKNOWN = object()


def _scope_nodes(scope: ast.AST):
    """Every node inside `scope`, without descending into nested function scopes."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(node))


def _scope_bindings(scope: ast.AST) -> List[Tuple[int, str, Any]]:
    """(line, name, value) bound directly in this scope: literal parameter
    defaults and assignments. A non-literal assignment binds _UNKNOWN, so a
    stale literal from an earlier line is never reused for it."""
    out: List[Tuple[int, str, Any]] = []
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        positional = list(getattr(scope.args, "posonlyargs", [])) + list(scope.args.args)
        pairs = list(zip(positional[len(positional) - len(scope.args.defaults):], scope.args.defaults))
        pairs += [(a, d) for a, d in zip(scope.args.kwonlyargs, scope.args.kw_defaults) if d is not None]
        for arg, default in pairs:
            ok, value = _literal_value(default)
            out.append((scope.lineno, arg.arg, value if ok else _UNKNOWN))
    for node in _scope_nodes(scope):
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        else:
            continue
        ok, value = _literal_value(node.value)
        for target in targets:
            if isinstance(target, ast.Name):
                out.append((node.lineno, target.id, value if ok else _UNKNOWN))
            elif isinstance(target, (ast.Tuple, ast.List)):
                names = [e.id if isinstance(e, ast.Name) else None for e in target.elts]
                unpacked = ok and isinstance(value, (tuple, list)) and len(value) == len(names)
                for i, name in enumerate(names):
                    if name is not None:
                        out.append((node.lineno, name, value[i] if unpacked else _UNKNOWN))
    return sorted(out, key=lambda b: b[0])


def _resolve_literal(node: ast.AST, envs: List[List[Tuple[int, str, Any]]], lineno: int) -> Tuple[bool, Any]:
    """A literal, or a name bound to one before `lineno` in the nearest enclosing scope."""
    ok, value = _literal_value(node)
    if ok:
        return True, value
    if not isinstance(node, ast.Name):
        return False, None
    for env in envs:
        prior = [v for ln, name, v in env if name == node.id and ln <= lineno]
        if prior:
            return (False, None) if prior[-1] is _UNKNOWN else (True, prior[-1])
    return False, None


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _jit_params(tree: ast.AST, name: str) -> List[str]:
    """Parameter names of the @triton.jit kernel, for positional launches."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name and _is_jit(node):
            return [a.arg for a in node.args.args]
    return []


def _launch_sites(tree: ast.AST) -> Dict[str, Dict[str, Any]]:
    """Literal kwargs + options of each kernel's first `kernel[grid](...)` launch, by name.

    Positional args hold their slot (None when unresolvable) so they line up
    with the kernel's parameters.
    """
    module_env = _scope_bindings(tree)
    scopes: List[ast.AST] = [tree] + sorted(
        (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
        key=lambda n: n.lineno)
    sites: Dict[str, Dict[str, Any]] = {}
    for scope in scopes:
        envs = [module_env] if scope is tree else [_scope_bindings(scope), module_env]
        calls = [n for n in _scope_nodes(scope)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript)]
        for node in sorted(calls, key=lambda c: c.lineno):
            target = node.func.value
            if not isinstance(target, ast.Name) or target.id in sites:
                continue
            constexprs: Dict[str, Any] = {}
            options: Dict[str, int] = {}
            for kw in node.keywords:
                ok, value = (False, None) if kw.arg is None else _resolve_literal(kw.value, envs, node.lineno)
                if not ok:
                    continue
                if kw.arg in ("num_warps", "num_stages"):
                    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                        options[kw.arg] = value
                else:
                    constexprs[kw.arg] = value
            positional = [_resolve_literal(a, envs, node.lineno)[1] for a in node.args]
            sites[target.id] = {
                "constexprs": constexprs,
                "positional": positional,
                "options": options,
            }
    return sites


def _first_autotune_config(tree: ast.AST) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """(constexprs, launch options) from the first literal triton.Config(...)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("Config") and node.args and isinstance(node.args[0], ast.Dict):
            try:
                constexprs = ast.literal_eval(node.args[0])
            except ValueError:
                continue
            options = {}
            for kw in node.keywords:
                if kw.arg in ("num_warps", "num_stages"):
                    try:
                        options[kw.arg] = int(ast.literal_eval(kw.value))
                    except ValueError:
                        pass
            return constexprs, options
    return {}, {}
