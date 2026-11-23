from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

from .attribution import install_path_attribution_patch


EXPECTED_CHECKPOINT_SHA256 = "7e99b13d90f0c0fbff42304eaa92ae4e35316034bec7847c71dcc4dc0e568e08"
EXPECTED_PARAMETER_COUNT = 103622

DEFAULT_NBFNET_ROOT = str(next(p for p in __import__('pathlib').Path(__file__).resolve().parents if (p / 'RELEASE_MANIFEST.json').is_file()) / 'backbone')
DEFAULT_RCDGATE_ROOT = str(next(p for p in __import__('pathlib').Path(__file__).resolve().parents if (p / 'RELEASE_MANIFEST.json').is_file()) / '.')
DEFAULT_CHECKPOINT = str(next(p for p in __import__('pathlib').Path(__file__).resolve().parents if (p / 'RELEASE_MANIFEST.json').is_file()) / 'results/seed42/best_valid_adr_mrr.pth')
DEFAULT_CONFIG = str(next(p for p in __import__('pathlib').Path(__file__).resolve().parents if (p / 'RELEASE_MANIFEST.json').is_file()) / 'results/seed42/config_snapshot.yaml')


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _add_import_roots(nbfnet_root: Path, rcdgate_root: Path) -> None:
    for path in (nbfnet_root, nbfnet_root / "script", rcdgate_root):
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def _safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_frozen_rcdgate(
    nbfnet_root=DEFAULT_NBFNET_ROOT,
    rcdgate_root=DEFAULT_RCDGATE_ROOT,
    checkpoint_path=DEFAULT_CHECKPOINT,
    config_path=DEFAULT_CONFIG,
    device="cuda",
):
    nbfnet_root = Path(nbfnet_root).resolve()
    rcdgate_root = Path(rcdgate_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    config_path = Path(config_path).resolve()
    for path in (nbfnet_root, rcdgate_root, checkpoint_path, config_path):
        if not path.exists():
            raise FileNotFoundError(path)
    actual_sha256 = file_sha256(checkpoint_path)
    if actual_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint SHA256 mismatch: expected {EXPECTED_CHECKPOINT_SHA256}, got {actual_sha256}"
        )
    _add_import_roots(nbfnet_root, rcdgate_root)
    from adr_rcdgate.config import load_config
    from adr_rcdgate.overlay import construct_model

    config = load_config(config_path)
    model = construct_model(nbfnet_root, True, config.model)
    checkpoint = _safe_torch_load(checkpoint_path)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError("checkpoint does not contain a model state_dict")
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict load mismatch: {incompatible}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"parameter count mismatch: expected {EXPECTED_PARAMETER_COUNT}, got {parameter_count}"
        )
    install_path_attribution_patch(type(model))
    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()
    return model, config, {
        "checkpoint_sha256": actual_sha256,
        "checkpoint_match": True,
        "strict_load": "ALL_KEYS_MATCHED",
        "parameter_count": parameter_count,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_valid_MRR": checkpoint.get("valid_MRR"),
        "device": str(target_device),
    }
