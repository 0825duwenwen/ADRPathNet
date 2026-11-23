"""Read-only path attribution utilities for the frozen NBFNet-RCDGate."""

from .attribution import (
    attribute_query,
    audit_edge_weight_independence,
    install_path_attribution_patch,
    ordinary_query_score,
)
from .graph import build_train_only_graph
from .loader import load_frozen_rcdgate
from .manifest import load_preflight_summary, load_query_manifest, resolve_queries

__all__ = [
    "attribute_query",
    "audit_edge_weight_independence",
    "build_train_only_graph",
    "install_path_attribution_patch",
    "load_frozen_rcdgate",
    "load_preflight_summary",
    "load_query_manifest",
    "ordinary_query_score",
    "resolve_queries",
]

__version__ = "0.2.1"
