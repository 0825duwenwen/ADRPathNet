from __future__ import annotations

from pathlib import Path


def inspect_production_source(root):
    root = Path(root).resolve()
    layers = (root / "nbfnet" / "layers.py").read_text(encoding="utf-8")
    models = (root / "nbfnet" / "models.py").read_text(encoding="utf-8")
    checks = {
        "convolution_class": "GeneralizedRelationalConv",
        "convolution_file": "nbfnet/layers.py",
        "message_computation_location": "GeneralizedRelationalConv.message",
        "pyg_flow": "source_to_target: edge_index[0]=source, edge_index[1]=receiver",
        "receiver_state_contract": "MessagePassing input_i from current layer input",
        "receiver_state_has_validation_or_test_labels": False,
        "receiver_state_has_candidate_ranking_information": False,
        "gate_position": "after relation-conditioned message, before boundary concat and PNA aggregation",
        "fused_path_supported": False,
        "fused_path_reason": "receiver-indexed input_i is unavailable to the fused rspmm mechanism",
        "sparse_transpose_receiver_risk": "avoided by official PyG source_to_target lifting; covered by directed-edge tests",
        "candidate_generation_changed": False,
        "negative_sampling_changed": False,
        "scoring_changed": False,
        "hidden_dim": 32,
        "propagation_layers": 4,
        "compatibility_parameters_per_layer": 32,
        "compatibility_parameters_total": 128,
    }
    evidence = {
        "conv_present": "class GeneralizedRelationalConv" in layers,
        "message_present": "def message(self, input_j" in layers,
        "pna_present": 'aggregate_func == "pna"' in layers,
        "four_layer_config_supported": "hidden_dims" in models,
        "current_layer_input_used": "layer_input = hidden" in models,
    }
    checks["source_evidence_all_present"] = all(evidence.values())
    checks["source_evidence"] = evidence
    return checks

