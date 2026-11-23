#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from adr_rcdgate_path.attribution import (
    attribute_query,
    finite_difference_direction_check,
    ordinary_query_score,
)
from adr_rcdgate_path.audit import (
    model_parameter_gradients_clear,
    runtime_environment,
    score_invariance,
    write_json,
)
from adr_rcdgate_path.exporter import build_path_records, write_path_csv, write_path_json
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
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_PARAMETER_COUNT,
    load_frozen_rcdgate,
    model_state_sha256,
)


DEFAULT_DATA_DIR = str(next(p for p in __import__('pathlib').Path(__file__).resolve().parents if (p / 'RELEASE_MANIFEST.json').is_file()) / 'data/DPIKG')
DEFAULT_OUTPUT_DIR = str(next(p for p in __import__('pathlib').Path(__file__).resolve().parents if (p / 'RELEASE_MANIFEST.json').is_file()) / 'results/path_preflight')
SMOKE_QUERIES = [
    {
        "query_id": "QUERY_1",
        "drug_name": "药品_盐酸左氧氟沙星胶囊",
        "relation_name": "引起",
        "adr_name": "腹胀",
    },
    {
        "query_id": "QUERY_2",
        "drug_name": "药品_盐酸环丙沙星片",
        "relation_name": "引起",
        "adr_name": "瘙痒",
    },
]


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
    parser = argparse.ArgumentParser(description="Stage 3A-0C frozen RCDGate runtime preflight")
    parser.add_argument("--nbfnet-root", default=DEFAULT_NBFNET_ROOT)
    parser.add_argument("--rcdgate-root", default=DEFAULT_RCDGATE_ROOT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-finite-difference", action="store_true")
    return parser.parse_args()


def query_tensor(query, graph, device):
    missing = [
        token
        for token in (query["drug_name"], query["adr_name"])
        if token not in graph.entity2id
    ]
    if query["relation_name"] not in graph.relation2id:
        missing.append(query["relation_name"])
    if missing:
        raise KeyError(f"smoke query tokens absent from frozen mappings: {missing}")
    h = int(graph.entity2id[query["drug_name"]])
    t = int(graph.entity2id[query["adr_name"]])
    r = int(graph.relation2id[query["relation_name"]])
    return torch.tensor([[h, t, r]], dtype=torch.long, device=device), h, t, r


def empty_summary():
    return {
        "CHECKPOINT_SHA256": "NOT_RUN",
        "CHECKPOINT_MATCH": False,
        "STRICT_LOAD": "NOT_RUN",
        "PARAMETER_COUNT": "NOT_RUN",
        "TRAIN_FORWARD_TRIPLES": "NOT_RUN",
        "TRAIN_GRAPH_EDGES_WITH_INVERSE": "NOT_RUN",
        "EDGE_WEIGHT_LAYER_COUNT": "NOT_RUN",
        "ALL_EDGE_WEIGHTS_LEAF": False,
        "EDGE_WEIGHT_AUTOGRAD_INDEPENDENT": False,
        "queries": [
            {
                "query_id": query["query_id"],
                "drug_name": query["drug_name"],
                "adr_name": query["adr_name"],
                "normal_score": "NOT_RUN",
                "path_mode_score": "NOT_RUN",
                "abs_score_diff": "NOT_RUN",
                "path_count": "NOT_RUN",
            }
            for query in SMOKE_QUERIES
        ],
        "PATH_EDGE_COUNT": "NOT_RUN",
        "PATH_EDGE_NOT_IN_TRAIN": "NOT_RUN",
        "VALID_EDGES_IN_REASONING_GRAPH": "NOT_RUN",
        "TEST_EDGES_IN_REASONING_GRAPH": "NOT_RUN",
        "MODEL_PARAMETER_CHANGED": "NOT_RUN",
        "PATH_METHOD": "SIGNED_EDGE_GRADIENT_BEAM_AVERAGE_LENGTH",
        "PATH_GRADIENT_PASSES_RCDGATE": False,
        "STAGE_3A0C_STATUS": "FAIL",
        "READY_FOR_STAGE_3A1": False,
    }


def print_console(summary):
    def display(value):
        if isinstance(value, bool):
            return str(value).upper()
        if isinstance(value, float):
            return f"{value:.12g}"
        return value

    print("===== STAGE 3A-0C RUNTIME PREFLIGHT =====")
    for key in (
        "CHECKPOINT_SHA256",
        "CHECKPOINT_MATCH",
        "STRICT_LOAD",
        "PARAMETER_COUNT",
        "TRAIN_FORWARD_TRIPLES",
        "TRAIN_GRAPH_EDGES_WITH_INVERSE",
        "EDGE_WEIGHT_LAYER_COUNT",
        "ALL_EDGE_WEIGHTS_LEAF",
        "EDGE_WEIGHT_AUTOGRAD_INDEPENDENT",
    ):
        print(f"{key}={display(summary[key])}")
    for query in summary["queries"]:
        print(f"{query['query_id']}={query['drug_name']} | 引起 | {query['adr_name']}")
        print(f"NORMAL_SCORE={display(query['normal_score'])}")
        print(f"PATH_MODE_SCORE={display(query['path_mode_score'])}")
        print(f"ABS_SCORE_DIFF={display(query['abs_score_diff'])}")
        print(f"PATH_COUNT={query['path_count']}")
    for key in (
        "PATH_EDGE_COUNT",
        "PATH_EDGE_NOT_IN_TRAIN",
        "VALID_EDGES_IN_REASONING_GRAPH",
        "TEST_EDGES_IN_REASONING_GRAPH",
        "MODEL_PARAMETER_CHANGED",
        "PATH_METHOD",
        "PATH_GRADIENT_PASSES_RCDGATE",
        "STAGE_3A0C_STATUS",
        "READY_FOR_STAGE_3A1",
    ):
        print(f"{key}={display(summary[key])}")


def run(args, output: Path):
    environment = runtime_environment()
    write_json(output / "runtime_environment.json", environment)
    graph = build_train_only_graph(args.data_dir)
    leakage = audit_split_leakage(graph)
    graph_audit = {
        "graph_source": "EXPLICIT_FROZEN_TRAIN_SPLIT",
        "processed_data_pt_used": False,
        "train_file": str(graph.train_file),
        "train_forward_triples": len(graph.forward_triples),
        "train_graph_edges_with_inverse": int(graph.edge_index.shape[1]),
        "num_nodes": graph.num_nodes,
        "num_forward_relations": graph.num_forward_relations,
        "inverse_relation_encoding": "inverse_id = forward_id + num_forward_relations",
        **leakage,
    }
    write_json(output / "graph_audit.json", graph_audit)

    model, config, checkpoint_audit = load_frozen_rcdgate(
        nbfnet_root=args.nbfnet_root,
        rcdgate_root=args.rcdgate_root,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
    )
    write_json(output / "checkpoint_audit.json", checkpoint_audit)
    device = next(model.parameters()).device
    data = graph.to_pyg(device)
    state_before = model_state_sha256(model)
    entity_names = {identifier: name for name, identifier in graph.entity2id.items()}
    relation_names = relation_display_map(graph.relation2id)
    train_edges = graph.directed_edge_set

    score_rows = []
    gradient_rows = []
    all_path_records = []
    edge_audits = []
    for query in SMOKE_QUERIES:
        batch, head_id, tail_id, relation_id = query_tensor(query, graph, device)
        normal_score = float(ordinary_query_score(model, data, batch))
        with torch.enable_grad():
            attribution = attribute_query(model, data, batch)
        invariance = score_invariance(normal_score, float(attribution.score))
        records, edge_audit = build_path_records(
            query_id=query["query_id"],
            drug_name=query["drug_name"],
            adr_name=query["adr_name"],
            head_id=head_id,
            tail_id=tail_id,
            paths=attribution.paths,
            path_scores=attribution.path_scores,
            entity_names=entity_names,
            relation_names=relation_names,
            train_edge_set=train_edges,
            max_hops=len(model.layers),
        )
        finite_difference = None
        if not args.skip_finite_difference:
            finite_difference = finite_difference_direction_check(
                model, data, batch, attribution, layer_index=0
            )
        score_rows.append({"query_id": query["query_id"], **invariance})
        gradient_rows.append(
            {
                "query_id": query["query_id"],
                "edge_weight_audit": attribution.edge_weight_audit,
                "all_edge_gradients_finite": all(
                    torch.isfinite(value).all() for value in attribution.edge_gradients
                ),
                "gate_gradient_reachable": attribution.gate_gradient_reachable,
                "gate_gradient_finite": attribution.gate_gradient_finite,
                "finite_difference": finite_difference,
            }
        )
        edge_audits.append({"query_id": query["query_id"], **edge_audit})
        all_path_records.extend(records)

    state_after = model_state_sha256(model)
    model_parameter_changed = state_before != state_after
    write_json(output / "score_invariance.json", score_rows)
    write_json(
        output / "path_gradient_audit.json",
        {
            "method": "signed local edge gradient -> beam search -> average path score",
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "model_parameter_changed": model_parameter_changed,
            "parameter_gradients_populated": not model_parameter_gradients_clear(model),
            "queries": gradient_rows,
        },
    )
    write_path_json(output / "smoke_paths.json", all_path_records)
    write_path_csv(output / "smoke_paths.csv", all_path_records)
    aggregate_edge_audit = {
        "queries": edge_audits,
        "path_edge_count": sum(row["path_edge_count"] for row in edge_audits),
        "path_edge_not_in_train": sum(row["path_edge_not_in_train"] for row in edge_audits),
        "path_continuity_failures": sum(row["path_continuity_failures"] for row in edge_audits),
        "path_endpoint_failures": sum(row["path_endpoint_failures"] for row in edge_audits),
        "path_hop_limit_failures": sum(row["path_hop_limit_failures"] for row in edge_audits),
    }
    write_json(output / "path_edge_audit.json", aggregate_edge_audit)

    all_leaf = all(row["edge_weight_audit"]["all_leaf"] for row in gradient_rows)
    all_independent = all(
        row["edge_weight_audit"]["autograd_independent"] for row in gradient_rows
    )
    gate_path = all(
        row["gate_gradient_reachable"] and row["gate_gradient_finite"]
        for row in gradient_rows
    )
    finite_difference_ok = all(
        row["finite_difference"] is None or row["finite_difference"]["passed"]
        for row in gradient_rows
    )
    checks = {
        "checkpoint_match": checkpoint_audit["checkpoint_match"],
        "strict_load": checkpoint_audit["strict_load"] == "ALL_KEYS_MATCHED",
        "parameter_count": checkpoint_audit["parameter_count"] == EXPECTED_PARAMETER_COUNT,
        "train_count": len(graph.forward_triples) == EXPECTED_FORWARD_TRIPLES,
        "directed_edge_count": graph.edge_index.shape[1] == EXPECTED_DIRECTED_EDGES,
        "edge_layer_count": all(
            row["edge_weight_audit"]["layer_count"] == len(model.layers)
            for row in gradient_rows
        ),
        "edge_weights_leaf": all_leaf,
        "edge_weights_independent": all_independent,
        "finite_difference": finite_difference_ok,
        "score_invariance": all(row["passed"] for row in score_rows),
        "paths_present": all(row["path_count"] > 0 for row in edge_audits),
        "path_membership": aggregate_edge_audit["path_edge_not_in_train"] == 0,
        "path_continuity": aggregate_edge_audit["path_continuity_failures"] == 0,
        "path_endpoints": aggregate_edge_audit["path_endpoint_failures"] == 0,
        "path_hops": aggregate_edge_audit["path_hop_limit_failures"] == 0,
        "valid_leakage": leakage["valid_edges_in_reasoning_graph"] == 0,
        "test_leakage": leakage["test_edges_in_reasoning_graph"] == 0,
        "parameters_unchanged": not model_parameter_changed,
        "parameter_grads_clear": model_parameter_gradients_clear(model),
        "gradient_passes_rcdgate": gate_path,
    }
    passed = all(checks.values())
    summary = {
        "CHECKPOINT_SHA256": checkpoint_audit["checkpoint_sha256"],
        "CHECKPOINT_MATCH": checkpoint_audit["checkpoint_match"],
        "STRICT_LOAD": checkpoint_audit["strict_load"],
        "PARAMETER_COUNT": checkpoint_audit["parameter_count"],
        "TRAIN_FORWARD_TRIPLES": len(graph.forward_triples),
        "TRAIN_GRAPH_EDGES_WITH_INVERSE": int(graph.edge_index.shape[1]),
        "EDGE_WEIGHT_LAYER_COUNT": len(model.layers),
        "ALL_EDGE_WEIGHTS_LEAF": all_leaf,
        "EDGE_WEIGHT_AUTOGRAD_INDEPENDENT": all_independent,
        "queries": [
            {
                "query_id": query["query_id"],
                "drug_name": query["drug_name"],
                "adr_name": query["adr_name"],
                "normal_score": score["normal_score"],
                "path_mode_score": score["path_mode_score"],
                "abs_score_diff": score["abs_score_diff"],
                "path_count": edge["path_count"],
            }
            for query, score, edge in zip(SMOKE_QUERIES, score_rows, edge_audits)
        ],
        "PATH_EDGE_COUNT": aggregate_edge_audit["path_edge_count"],
        "PATH_EDGE_NOT_IN_TRAIN": aggregate_edge_audit["path_edge_not_in_train"],
        "VALID_EDGES_IN_REASONING_GRAPH": leakage["valid_edges_in_reasoning_graph"],
        "TEST_EDGES_IN_REASONING_GRAPH": leakage["test_edges_in_reasoning_graph"],
        "MODEL_PARAMETER_CHANGED": model_parameter_changed,
        "PATH_METHOD": "SIGNED_EDGE_GRADIENT_BEAM_AVERAGE_LENGTH",
        "PATH_GRADIENT_PASSES_RCDGATE": gate_path,
        "STAGE_3A0C_STATUS": "PASS" if passed else "FAIL",
        "READY_FOR_STAGE_3A1": passed,
        "checks": checks,
    }
    write_json(output / "stage3a0c_summary.json", summary)
    return summary


def main():
    args = parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output directory: {output}")
    output.mkdir(parents=True)
    log_path = output / "stage3a0c_console.log"
    summary = empty_summary()
    with log_path.open("w", encoding="utf-8") as log_stream:
        tee_out = Tee(sys.stdout, log_stream)
        tee_err = Tee(sys.stderr, log_stream)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            try:
                summary = run(args, output)
            except Exception as error:
                print(f"RUNTIME_ERROR={type(error).__name__}: {error}")
                traceback.print_exc()
                write_json(output / "stage3a0c_failure.json", {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "ready_for_stage_3a1": False,
                })
            print_console(summary)
    if summary["STAGE_3A0C_STATUS"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
