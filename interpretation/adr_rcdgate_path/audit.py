from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch


def write_json(path, value) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def runtime_environment() -> dict:
    result = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    try:
        import torch_geometric

        result["torch_geometric"] = torch_geometric.__version__
    except Exception as error:
        result["torch_geometric"] = None
        result["torch_geometric_import_error"] = repr(error)
    try:
        import torch_scatter

        result["torch_scatter"] = torch_scatter.__version__
    except Exception as error:
        result["torch_scatter"] = None
        result["torch_scatter_import_error"] = repr(error)
    return result


def score_invariance(normal_score: float, path_score: float, tolerance: float = 1e-6) -> dict:
    difference = abs(float(normal_score) - float(path_score))
    return {
        "normal_score": float(normal_score),
        "path_mode_score": float(path_score),
        "abs_score_diff": difference,
        "tolerance": tolerance,
        "passed": difference <= tolerance,
    }


def model_parameter_gradients_clear(model) -> bool:
    return all(parameter.grad is None for parameter in model.parameters())


def require_all_checks(checks: dict[str, bool]) -> None:
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(f"Stage 3A-0C failed checks: {failures}")
