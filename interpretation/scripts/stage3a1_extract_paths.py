#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from adr_rcdgate_path.attribution import attribute_query, ordinary_query_score
from adr_rcdgate_path.audit import model_parameter_gradients_clear, score_invariance, write_json
from adr_rcdgate_path.exporter import (
    build_formal_path_records,
    write_formal_path_csv,
    write_human_readable_paths,
    write_path_json,
    write_query_summary_csv,
)
from adr_rcdgate_path.graph import (
    EXPECTED_DIRECTED_EDGES,
    EXPECTED_FORWARD_TRIPLES,
    audit_split_leakage,
    build_train_only_graph,
    relation_display_map,
)
from adr_rcdgate_path.loader import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_NBFNET_ROOT,
    DEFAULT_RCDGATE_ROOT,
    EXPECTED_PARAMETER_COUNT,
    load_frozen_rcdgate,
    model_state_sha256,
)
from adr_rcdgate_path.manifest import (
    QueryEntityNotFound,
    load_preflight_summary,
    load_query_manifest,
    resolve_queries,
)

DEFAULT_DATA_DIR = str(next(p for p in __import__('pathlib').Path(__file__).resolve().parents if (p / 'RELEASE_MANIFEST.json').is_file()) / 'data/DPIKG')
DEFAULT_OUTPUT_DIR = str(next(p for p in __import__('pathlib').Path(__file__).resolve().parents if (p / 'RELEASE_MANIFEST.json').is_file()) / 'outputs/NBFNet-RCDGate-Path/stage3a1_12pair_seed42_v0.1.0')
PATH_METHOD = "SIGNED_EDGE_GRADIENT_BEAM_AVERAGE_LENGTH"
SCORE_INVARIANCE_TOLERANCE = 1e-5


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3A-1 frozen 12-pair RCDGate path extraction")
    parser.add_argument("--preflight-summary", required=True)
    parser.add_argument("--query-manifest", required=True)
    parser.add_argument("--nbfnet-root", default=DEFAULT_NBFNET_ROOT)
    parser.add_argument("--rcdgate-root", default=DEFAULT_RCDGATE_ROOT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def empty_summary(output_dir):
    return {
        "QUERY_COUNT": 12, "QUERY_RESOLVED_COUNT": 0,
        "CHECKPOINT_SHA256": "NOT_RUN", "CHECKPOINT_MATCH": False,
        "STRICT_LOAD": "NOT_RUN", "PARAMETER_COUNT": "NOT_RUN",
        "TRAIN_FORWARD_TRIPLES": "NOT_RUN", "TRAIN_GRAPH_EDGES_WITH_INVERSE": "NOT_RUN",
        "EDGE_WEIGHT_LAYER_COUNT": "NOT_RUN", "ALL_EDGE_WEIGHTS_LEAF": False,
        "EDGE_WEIGHT_AUTOGRAD_INDEPENDENT": False, "TOTAL_PATH_COUNT": "NOT_RUN",
        "POSITIVE_PATH_COUNT": "NOT_RUN", "NEGATIVE_PATH_COUNT": "NOT_RUN",
        "ZERO_PATH_COUNT": "NOT_RUN", "QUERY_WITH_PATH_COUNT": "NOT_RUN",
        "QUERY_WITHOUT_PATH_COUNT": "NOT_RUN", "PATH_EDGE_COUNT": "NOT_RUN",
        "PATH_EDGE_NOT_IN_TRAIN": "NOT_RUN", "BROKEN_PATH_COUNT": "NOT_RUN",
        "WRONG_START_COUNT": "NOT_RUN", "WRONG_END_COUNT": "NOT_RUN",
        "OVER_DEPTH_PATH_COUNT": "NOT_RUN", "VALID_EDGES_IN_REASONING_GRAPH": "NOT_RUN",
        "TEST_EDGES_IN_REASONING_GRAPH": "NOT_RUN",
        "SCORE_INVARIANCE_TOLERANCE": SCORE_INVARIANCE_TOLERANCE,
        "MAX_ABS_SCORE_DIFF": "NOT_RUN",
        "ALL_SCORE_DIFF_WITHIN_TOLERANCE": False, "MODEL_PARAMETER_CHANGED": "NOT_RUN",
        "PATH_METHOD": PATH_METHOD, "PATH_GRADIENT_PASSES_RCDGATE": False,
        "OUTPUT_DIR": str(output_dir), "STAGE_3A1_STATUS": "FAIL",
        "READY_FOR_PATH_CASE_SELECTION": False,
    }


def display(value):
    return str(value).upper() if isinstance(value, bool) else value


def print_console(summary):
    print("===== STAGE 3A-1 12-PAIR PATH EXTRACTION =====")
    fields = [
        "QUERY_COUNT", "QUERY_RESOLVED_COUNT", "CHECKPOINT_SHA256", "CHECKPOINT_MATCH",
        "STRICT_LOAD", "PARAMETER_COUNT", "TRAIN_FORWARD_TRIPLES",
        "TRAIN_GRAPH_EDGES_WITH_INVERSE", "EDGE_WEIGHT_LAYER_COUNT",
        "ALL_EDGE_WEIGHTS_LEAF", "EDGE_WEIGHT_AUTOGRAD_INDEPENDENT",
        "TOTAL_PATH_COUNT", "POSITIVE_PATH_COUNT", "NEGATIVE_PATH_COUNT",
        "ZERO_PATH_COUNT", "QUERY_WITH_PATH_COUNT", "QUERY_WITHOUT_PATH_COUNT",
        "PATH_EDGE_COUNT", "PATH_EDGE_NOT_IN_TRAIN", "BROKEN_PATH_COUNT",
        "WRONG_START_COUNT", "WRONG_END_COUNT", "OVER_DEPTH_PATH_COUNT",
        "VALID_EDGES_IN_REASONING_GRAPH", "TEST_EDGES_IN_REASONING_GRAPH",
        "SCORE_INVARIANCE_TOLERANCE", "MAX_ABS_SCORE_DIFF",
        "ALL_SCORE_DIFF_WITHIN_TOLERANCE", "MODEL_PARAMETER_CHANGED", "PATH_METHOD",
        "PATH_GRADIENT_PASSES_RCDGATE", "OUTPUT_DIR", "STAGE_3A1_STATUS",
        "READY_FOR_PATH_CASE_SELECTION",
    ]
    for field in fields:
        print(f"{field}={display(summary[field])}")


def resolved_payload(manifest, resolved):
    return {
        "manifest_version": manifest["manifest_version"], "frozen": True,
        "selection_policy": manifest["selection_policy"],
        "resolution_method": "EXACT_CANONICAL_LOOKUP_ONLY",
        "query_count": len(manifest["queries"]), "query_resolved_count": len(resolved),
        "queries": resolved,
    }


def run(args, output: Path, preflight: dict, manifest: dict):
    graph = build_train_only_graph(args.data_dir)
    leakage = audit_split_leakage(graph)
    resolved = resolve_queries(manifest, graph.entity2id, graph.relation2id)
    write_json(output / "queries_12_frozen_resolved.json", resolved_payload(manifest, resolved))
    graph_audit = {
        "graph_source": "EXPLICIT_FROZEN_TRAIN_SPLIT", "processed_data_pt_used": False,
        "all_true_used_as_reasoning_edges": False, "train_file": str(graph.train_file),
        "train_forward_triples": len(graph.forward_triples),
        "train_graph_edges_with_inverse": int(graph.edge_index.shape[1]),
        "num_nodes": graph.num_nodes, "num_forward_relations": graph.num_forward_relations,
        "inverse_relation_encoding": "inverse_id = forward_id + num_forward_relations",
        **leakage,
    }
    write_json(output / "graph_audit.json", graph_audit)
    model, _config, checkpoint_audit = load_frozen_rcdgate(
        nbfnet_root=args.nbfnet_root, rcdgate_root=args.rcdgate_root,
        checkpoint_path=args.checkpoint, config_path=args.config, device=args.device,
    )
    if len(model.layers) != 5:
        raise RuntimeError(f"frozen layer count mismatch: {len(model.layers)}")
    if model.path_topk != 10:
        raise RuntimeError(f"frozen path_topk must be 10, observed {model.path_topk}")
    write_json(output / "checkpoint_audit.json", {
        **checkpoint_audit, "num_layers": len(model.layers), "num_beam": model.num_beam,
        "path_topk": model.path_topk,
        "stage3a0c_preflight_gate": {
            field: preflight[field] for field in (
                "READY_FOR_STAGE_3A1", "STAGE_3A0C_STATUS", "CHECKPOINT_MATCH",
                "EDGE_WEIGHT_AUTOGRAD_INDEPENDENT", "PATH_GRADIENT_PASSES_RCDGATE",
            )
        },
    })
    device = next(model.parameters()).device
    data = graph.to_pyg(device)
    entity_names = {identifier: name for name, identifier in graph.entity2id.items()}
    relation_names = relation_display_map(graph.relation2id)
    train_edges = graph.directed_edge_set
    state_before = model_state_sha256(model)
    all_records, query_rows, query_audits, gradient_audits = [], [], [], []
    for query in resolved:
        batch = torch.tensor([[query["head_id"], query["tail_id"], query["relation_id"]]], dtype=torch.long, device=device)
        normal_score = float(ordinary_query_score(model, data, batch))
        with torch.enable_grad():
            attribution = attribute_query(model, data, batch)
        path_mode_score = float(attribution.score)
        score_audit = score_invariance(
            normal_score,
            path_mode_score,
            tolerance=SCORE_INVARIANCE_TOLERANCE,
        )
        records, path_audit = build_formal_path_records(
            query, normal_score, attribution.paths, attribution.path_scores, entity_names,
            relation_names, train_edges, len(model.layers),
        )
        positive = sum(record["contribution_sign"] == "POSITIVE" for record in records)
        negative = sum(record["contribution_sign"] == "NEGATIVE" for record in records)
        zero = sum(record["contribution_sign"] == "ZERO" for record in records)
        query_rows.append({
            "query_id": query["query_id"], "drug": query["drug_display"],
            "ADR": query["adr_display"], "discovery_rank": query["discovery_rank"],
            "model_score": normal_score, "normal_score": normal_score,
            "path_mode_score": path_mode_score, "abs_score_diff": score_audit["abs_score_diff"],
            "path_count": len(records), "positive_path_count": positive,
            "negative_path_count": negative, "zero_path_count": zero,
            "best_path_score": max((record["signed_path_score"] for record in records), default=None),
        })
        query_audits.append({
            **path_audit, "normal_score": normal_score, "path_mode_score": path_mode_score,
            "abs_score_diff": score_audit["abs_score_diff"],
            "score_within_tolerance": score_audit["passed"],
        })
        gradient_audits.append({
            "query_id": query["query_id"], "edge_weight_audit": attribution.edge_weight_audit,
            "all_edge_gradients_finite": all(torch.isfinite(value).all() for value in attribution.edge_gradients),
            "gate_gradient_reachable": attribution.gate_gradient_reachable,
            "gate_gradient_finite": attribution.gate_gradient_finite,
        })
        all_records.extend(records)
    write_path_json(output / "paths_all.json", all_records)
    write_formal_path_csv(output / "paths_all.csv", all_records)
    write_query_summary_csv(output / "query_summary.csv", query_rows)
    write_human_readable_paths(
        output / "paths_human_readable.txt", resolved, all_records, query_rows
    )
    integrity = {
        "query_audits": query_audits,
        "path_edge_count": sum(row["path_edge_count"] for row in query_audits),
        "path_edge_not_in_train": sum(row["path_edge_not_in_train"] for row in query_audits),
        "broken_path_count": sum(row["broken_path_count"] for row in query_audits),
        "wrong_start_count": sum(row["wrong_start_count"] for row in query_audits),
        "wrong_end_count": sum(row["wrong_end_count"] for row in query_audits),
        "over_depth_path_count": sum(row["over_depth_path_count"] for row in query_audits),
    }
    write_json(output / "path_integrity_audit.json", integrity)
    state_after = model_state_sha256(model)
    parameter_changed = state_before != state_after
    write_json(output / "parameter_integrity.json", {
        "model_state_sha256_before": state_before, "model_state_sha256_after": state_after,
        "model_parameter_changed": parameter_changed,
        "parameter_gradients_populated": not model_parameter_gradients_clear(model),
        "optimizer_created": False, "training_performed": False,
    })
    all_leaf = all(row["edge_weight_audit"]["all_leaf"] for row in gradient_audits)
    all_independent = all(row["edge_weight_audit"]["autograd_independent"] for row in gradient_audits)
    gradient_passes_gate = all(row["gate_gradient_reachable"] and row["gate_gradient_finite"] for row in gradient_audits)
    all_score_ok = all(row["score_within_tolerance"] for row in query_audits)
    max_abs_score_diff = max(row["abs_score_diff"] for row in query_audits)
    positive_total = sum(row["positive_path_count"] for row in query_rows)
    negative_total = sum(row["negative_path_count"] for row in query_rows)
    zero_total = sum(row["zero_path_count"] for row in query_rows)
    checks = {
        "preflight_gate": True, "query_count": len(manifest["queries"]) == 12,
        "query_resolved_count": len(resolved) == 12,
        "checkpoint_match": checkpoint_audit["checkpoint_match"],
        "strict_load": checkpoint_audit["strict_load"] == "ALL_KEYS_MATCHED",
        "parameter_count": checkpoint_audit["parameter_count"] == EXPECTED_PARAMETER_COUNT,
        "train_forward_count": len(graph.forward_triples) == EXPECTED_FORWARD_TRIPLES,
        "train_directed_count": graph.edge_index.shape[1] == EXPECTED_DIRECTED_EDGES,
        "five_layers": len(model.layers) == 5, "path_topk_frozen": model.path_topk == 10,
        "edge_weights_leaf": all_leaf, "edge_weights_independent": all_independent,
        "score_tolerance": all_score_ok,
        "path_edge_membership": integrity["path_edge_not_in_train"] == 0,
        "path_continuity": integrity["broken_path_count"] == 0,
        "path_start": integrity["wrong_start_count"] == 0,
        "path_end": integrity["wrong_end_count"] == 0,
        "path_depth": integrity["over_depth_path_count"] == 0,
        "valid_leakage": leakage["valid_edges_in_reasoning_graph"] == 0,
        "test_leakage": leakage["test_edges_in_reasoning_graph"] == 0,
        "parameters_unchanged": not parameter_changed,
        "parameter_grads_clear": model_parameter_gradients_clear(model),
        "gradient_passes_rcdgate": gradient_passes_gate,
    }
    passed = all(checks.values())
    summary = {
        "QUERY_COUNT": len(manifest["queries"]), "QUERY_RESOLVED_COUNT": len(resolved),
        "CHECKPOINT_SHA256": checkpoint_audit["checkpoint_sha256"],
        "CHECKPOINT_MATCH": checkpoint_audit["checkpoint_match"],
        "STRICT_LOAD": checkpoint_audit["strict_load"], "PARAMETER_COUNT": checkpoint_audit["parameter_count"],
        "TRAIN_FORWARD_TRIPLES": len(graph.forward_triples),
        "TRAIN_GRAPH_EDGES_WITH_INVERSE": int(graph.edge_index.shape[1]),
        "EDGE_WEIGHT_LAYER_COUNT": len(model.layers), "ALL_EDGE_WEIGHTS_LEAF": all_leaf,
        "EDGE_WEIGHT_AUTOGRAD_INDEPENDENT": all_independent,
        "TOTAL_PATH_COUNT": len(all_records), "POSITIVE_PATH_COUNT": positive_total,
        "NEGATIVE_PATH_COUNT": negative_total, "ZERO_PATH_COUNT": zero_total,
        "QUERY_WITH_PATH_COUNT": sum(row["path_count"] > 0 for row in query_rows),
        "QUERY_WITHOUT_PATH_COUNT": sum(row["path_count"] == 0 for row in query_rows),
        "PATH_EDGE_COUNT": integrity["path_edge_count"],
        "PATH_EDGE_NOT_IN_TRAIN": integrity["path_edge_not_in_train"],
        "BROKEN_PATH_COUNT": integrity["broken_path_count"],
        "WRONG_START_COUNT": integrity["wrong_start_count"],
        "WRONG_END_COUNT": integrity["wrong_end_count"],
        "OVER_DEPTH_PATH_COUNT": integrity["over_depth_path_count"],
        "VALID_EDGES_IN_REASONING_GRAPH": leakage["valid_edges_in_reasoning_graph"],
        "TEST_EDGES_IN_REASONING_GRAPH": leakage["test_edges_in_reasoning_graph"],
        "SCORE_INVARIANCE_TOLERANCE": SCORE_INVARIANCE_TOLERANCE,
        "MAX_ABS_SCORE_DIFF": max_abs_score_diff,
        "ALL_SCORE_DIFF_WITHIN_TOLERANCE": all_score_ok,
        "MODEL_PARAMETER_CHANGED": parameter_changed, "PATH_METHOD": PATH_METHOD,
        "PATH_GRADIENT_PASSES_RCDGATE": gradient_passes_gate, "OUTPUT_DIR": str(output),
        "STAGE_3A1_STATUS": "PASS" if passed else "FAIL",
        "READY_FOR_PATH_CASE_SELECTION": passed, "num_beam": model.num_beam,
        "path_topk": model.path_topk, "checks": checks,
        "query_summary": query_rows, "gradient_audits": gradient_audits,
    }
    write_json(output / "stage3a1_summary.json", summary)
    return summary


def main():
    args = parse_args()
    # Gate before output creation and before any graph/model work.
    preflight = load_preflight_summary(args.preflight_summary)
    manifest = load_query_manifest(args.query_manifest)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output directory: {output}")
    output.mkdir(parents=True)
    summary = empty_summary(output)
    with (output / "stage3a1_console.log").open("w", encoding="utf-8") as log_stream:
        with redirect_stdout(Tee(sys.stdout, log_stream)), redirect_stderr(Tee(sys.stderr, log_stream)):
            try:
                summary = run(args, output, preflight, manifest)
            except QueryEntityNotFound as error:
                print(f"QUERY_ENTITY_NOT_FOUND={error}")
                write_json(output / "stage3a1_failure.json", {
                    "error_code": error.code, "error": str(error), "status": "FAIL",
                })
            except Exception as error:
                print(f"RUNTIME_ERROR={type(error).__name__}: {error}")
                traceback.print_exc()
                write_json(output / "stage3a1_failure.json", {
                    "error_type": type(error).__name__, "error": str(error), "status": "FAIL",
                })
            print_console(summary)
    if summary["STAGE_3A1_STATUS"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
