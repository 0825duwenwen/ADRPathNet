#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from adr_rcdgate_path.attribution import attribute_query, ordinary_query_score
from adr_rcdgate_path.graph import audit_split_leakage, build_train_only_graph, relation_display_map
from adr_rcdgate_path.loader import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_NBFNET_ROOT,
    DEFAULT_RCDGATE_ROOT,
    EXPECTED_PARAMETER_COUNT,
    load_frozen_rcdgate,
    model_state_sha256,
)

VERSION = "0.1.0"
PATH_METHOD = "SIGNED_EDGE_GRADIENT_BEAM_AVERAGE_LENGTH"
EXPECTED_CHECKPOINT_SHA256 = "7e99b13d90f0c0fbff42304eaa92ae4e35316034bec7847c71dcc4dc0e568e08"
EXPECTED_FORWARD_TRIPLES = 65148
EXPECTED_DIRECTED_EDGES = 130296
EXPECTED_NUM_LAYERS = 5
DISPLAY_EDGE_K = 8
RECONSTRUCTION_TOLERANCE = 1e-5

DEFAULT_DATA_DIR = str(next(p for p in __import__('pathlib').Path(__file__).resolve().parents if (p / 'RELEASE_MANIFEST.json').is_file()) / 'data/DPIKG')
DEFAULT_PATH_DIR = (
    str(next(p for p in __import__('pathlib').Path(__file__).resolve().parents if (p / 'RELEASE_MANIFEST.json').is_file()) / 'results/paths')
)
DEFAULT_OUTPUT_DIR = (
    str(next(p for p in __import__('pathlib').Path(__file__).resolve().parents if (p / 'RELEASE_MANIFEST.json').is_file()) / 'outputs/NBFNet-RCDGate-Path/stage3a1b_local_edge_attribution_seed42_v0.1.0')
)

TARGET_QUERIES = ("Q04", "Q11", "Q12")
REPRESENTATIVE_RANK = {"Q04": 1, "Q11": 1, "Q12": 3}

# Frozen representative-path identities.  Triples use reasoning-graph direction:
# (source_id, relation_id, target_id)
EXPECTED_REPRESENTATIVE_TRIPLES = {
    "Q04": [
        (10585, 0, 3629),
        (3629, 3, 6678),
        (6678, 2, 1306),
    ],
    "Q11": [
        (10690, 2, 55),
        (55, 5, 11134),
        (11134, 2, 801),
    ],
    "Q12": [
        (10690, 2, 55),
        (55, 5, 8856),
        (8856, 2, 787),
    ],
}

# Fixed paper-facing mappings already approved for Fig. 3.
EN_ENTITY = {
    "药品_盐酸左氧氟沙星胶囊": "Levofloxacin Hydrochloride Capsules",
    "组成成分_盐酸左氧氟沙星": "Levofloxacin Hydrochloride",
    "药品_同林": "Tonglin",
    "不良反应_过敏反应": "Allergic reaction",
    "药品_盐酸环丙沙星片": "Ciprofloxacin Hydrochloride Tablets",
    "不良反应_中枢神经系统反应": "Central nervous system reaction",
    "药品_索多泰": "Suoduotai",
    "不良反应_皮疹": "Rash",
    "药品_来立信": "Lailixin",
    "不良反应_瘙痒": "Pruritus",
}

EN_RELATION = {
    "包含": "Contains",
    "包含^-1": "Contains^-1",
    "累及": "Affects",
    "累及^-1": "Affects^-1",
    "引起": "Causes",
    "引起^-1": "Causes^-1",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer CSV columns for empty rows: {path}")
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_mapping(raw: dict, name: str) -> dict[str, int]:
    out = {}
    for key, value in raw.items():
        if isinstance(value, int):
            out[str(key)] = int(value)
        elif isinstance(key, str) and key.isdigit():
            out[str(value)] = int(key)
        else:
            raise ValueError(f"Unsupported {name} mapping entry: {key!r}: {value!r}")
    return out


def load_mappings(data_dir: Path) -> tuple[dict[str, int], dict[str, int]]:
    entity2id = normalize_mapping(
        json.loads((data_dir / "entity2id.json").read_text(encoding="utf-8-sig")),
        "entity2id",
    )
    relation2id = normalize_mapping(
        json.loads((data_dir / "relation2id.json").read_text(encoding="utf-8-sig")),
        "relation2id",
    )
    return entity2id, relation2id


def _entity_token(token: str, entity2id: dict[str, int], valid_ids: set[int]) -> int:
    if token in entity2id:
        return int(entity2id[token])
    try:
        value = int(token)
    except ValueError as exc:
        raise ValueError(f"Unknown entity token {token!r}") from exc
    if value not in valid_ids:
        raise ValueError(f"Entity id out of mapping: {value}")
    return value


def _relation_token(token: str, relation2id: dict[str, int], valid_ids: set[int]) -> int:
    if token in relation2id:
        return int(relation2id[token])
    try:
        value = int(token)
    except ValueError as exc:
        raise ValueError(f"Unknown relation token {token!r}") from exc
    if value not in valid_ids:
        raise ValueError(f"Relation id out of mapping: {value}")
    return value


def _split_parts(line: str) -> list[str]:
    parts = line.rstrip("\n\r").split("\t")
    if len(parts) != 3:
        parts = line.split()
    if len(parts) != 3:
        raise ValueError(f"Expected exactly 3 columns, got {len(parts)}: {line!r}")
    return parts


def _try_parse_split(
    lines: list[str],
    layout: str,
    entity2id: dict[str, int],
    relation2id: dict[str, int],
) -> list[tuple[int, int, int]]:
    entity_ids = set(entity2id.values())
    relation_ids = set(relation2id.values())
    rows = []
    for line in lines:
        p = _split_parts(line)
        if layout == "HRT":
            h = _entity_token(p[0], entity2id, entity_ids)
            r = _relation_token(p[1], relation2id, relation_ids)
            t = _entity_token(p[2], entity2id, entity_ids)
        elif layout == "HTR":
            h = _entity_token(p[0], entity2id, entity_ids)
            t = _entity_token(p[1], entity2id, entity_ids)
            r = _relation_token(p[2], relation2id, relation_ids)
        else:
            raise ValueError(layout)
        rows.append((h, r, t))
    return rows


def read_split(
    path: Path,
    entity2id: dict[str, int],
    relation2id: dict[str, int],
) -> tuple[set[tuple[int, int, int]], str]:
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    successes = {}
    for layout in ("HRT", "HTR"):
        try:
            successes[layout] = _try_parse_split(lines, layout, entity2id, relation2id)
        except Exception:
            pass
    if len(successes) != 1:
        raise RuntimeError(
            f"Could not uniquely infer split layout for {path}; successful layouts={sorted(successes)}"
        )
    layout, rows = next(iter(successes.items()))
    return set(rows), layout


def parse_args():
    p = argparse.ArgumentParser(
        description="Read-only recovery of Q04/Q11/Q12 layer-wise edge gradients and Fig.3 local attributed subgraphs"
    )
    p.add_argument("--nbfnet-root", default=DEFAULT_NBFNET_ROOT)
    p.add_argument("--rcdgate-root", default=DEFAULT_RCDGATE_ROOT)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--path-dir", default=DEFAULT_PATH_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def path_triples(record: dict) -> list[tuple[int, int, int]]:
    nodes = [int(v) for v in record["node_ids"]]
    rels = [int(v) for v in record["relation_ids"]]
    if len(nodes) != len(rels) + 1:
        raise RuntimeError(
            f"{record.get('query_id')} path rank={record.get('path_rank')} has invalid node/relation lengths"
        )
    return [(nodes[i], rels[i], nodes[i + 1]) for i in range(len(rels))]


def canonical_forward(
    source: int, relation: int, target: int, num_forward_relations: int
) -> tuple[tuple[int, int, int], str]:
    if 0 <= relation < num_forward_relations:
        return (source, relation, target), "forward_train_edge"
    if num_forward_relations <= relation < 2 * num_forward_relations:
        return (
            target,
            relation - num_forward_relations,
            source,
        ), "inverse_of_forward_train_edge"
    raise ValueError(f"Unexpected reasoning relation id {relation}")


def entity_display(name: str) -> tuple[str, str]:
    if name in EN_ENTITY:
        return EN_ENTITY[name], "FROZEN_APPROVED_MAPPING"
    return "", "NEEDS_MANUAL_LABEL_REVIEW"


def relation_en(name: str) -> str:
    return EN_RELATION.get(name, "")


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    path_dir = Path(args.path_dir)
    output = Path(args.output_dir)

    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)

    required_inputs = [
        checkpoint_path,
        config_path,
        data_dir / "entity2id.json",
        data_dir / "relation2id.json",
        data_dir / "train_id.txt",
        data_dir / "valid_id.txt",
        data_dir / "test_id.txt",
        path_dir / "paths_all.json",
        path_dir / "paths_all.csv",
        path_dir / "queries_12_frozen_resolved.json",
        path_dir / "stage3a1_summary.json",
        path_dir / "graph_audit.json",
        path_dir / "path_integrity_audit.json",
        PACKAGE_ROOT / "adr_rcdgate_path" / "attribution.py",
    ]
    for p in required_inputs:
        if not p.exists():
            raise FileNotFoundError(p)

    # ---------- frozen artifact audit ----------
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Frozen checkpoint SHA mismatch: {checkpoint_sha} != {EXPECTED_CHECKPOINT_SHA256}"
        )

    old_records = json.loads((path_dir / "paths_all.json").read_text(encoding="utf-8"))
    if len(old_records) != 120:
        raise RuntimeError(f"Expected 120 frozen path records, observed {len(old_records)}")

    with (path_dir / "paths_all.csv").open("r", encoding="utf-8-sig", newline="") as f:
        old_csv_fields = next(csv.reader(f))
    gradient_tokens = ("gradient", "edge_attribution", "edge_gradient")
    json_has_gradient = any(
        any(token in key.lower() for token in gradient_tokens)
        for record in old_records
        for key in record.keys()
    )
    csv_has_gradient = any(
        any(token in field.lower() for token in gradient_tokens)
        for field in old_csv_fields
    )
    existing_export_has_edge_gradient = bool(json_has_gradient or csv_has_gradient)

    resolved_payload = json.loads(
        (path_dir / "queries_12_frozen_resolved.json").read_text(encoding="utf-8")
    )
    resolved_all = resolved_payload["queries"]
    resolved_by_qid = {q["query_id"]: q for q in resolved_all}
    for qid in TARGET_QUERIES:
        if qid not in resolved_by_qid:
            raise RuntimeError(f"Missing frozen resolved query {qid}")

    records_by_qid = {}
    for qid in TARGET_QUERIES:
        rows = [r for r in old_records if r["query_id"] == qid]
        rows.sort(key=lambda r: int(r["path_rank"]))
        if len(rows) != 10:
            raise RuntimeError(f"{qid}: expected 10 frozen paths, observed {len(rows)}")
        records_by_qid[qid] = rows

    representative_record = {}
    for qid in TARGET_QUERIES:
        rank = REPRESENTATIVE_RANK[qid]
        matches = [r for r in records_by_qid[qid] if int(r["path_rank"]) == rank]
        if len(matches) != 1:
            raise RuntimeError(f"{qid}: representative rank {rank} matched {len(matches)} records")
        record = matches[0]
        observed_triples = path_triples(record)
        if observed_triples != EXPECTED_REPRESENTATIVE_TRIPLES[qid]:
            raise RuntimeError(
                f"{qid}: frozen representative path identity mismatch\n"
                f"expected={EXPECTED_REPRESENTATIVE_TRIPLES[qid]}\nobserved={observed_triples}"
            )
        representative_record[qid] = record

    # ---------- graph and split audit ----------
    graph = build_train_only_graph(str(data_dir))
    leakage = audit_split_leakage(graph)
    if len(graph.forward_triples) != EXPECTED_FORWARD_TRIPLES:
        raise RuntimeError(
            f"Forward train count mismatch: {len(graph.forward_triples)} != {EXPECTED_FORWARD_TRIPLES}"
        )
    directed_edge_count = int(graph.edge_index.shape[1])
    if directed_edge_count != EXPECTED_DIRECTED_EDGES:
        raise RuntimeError(
            f"Directed reasoning edge count mismatch: {directed_edge_count} != {EXPECTED_DIRECTED_EDGES}"
        )

    src_cpu = graph.edge_index[0].cpu().tolist()
    dst_cpu = graph.edge_index[1].cpu().tolist()
    rel_cpu = graph.edge_type.cpu().tolist()
    directed_triples = [(int(s), int(r), int(t)) for s, r, t in zip(src_cpu, rel_cpu, dst_cpu)]
    counts = Counter(directed_triples)
    duplicate_types = sum(1 for v in counts.values() if v > 1)
    if duplicate_types:
        raise RuntimeError(f"Directed reasoning graph contains duplicate triples: {duplicate_types}")
    edge_index_by_triple = {triple: i for i, triple in enumerate(directed_triples)}

    entity2id, relation2id = load_mappings(data_dir)
    train_set, train_layout = read_split(data_dir / "train_id.txt", entity2id, relation2id)
    valid_set, valid_layout = read_split(data_dir / "valid_id.txt", entity2id, relation2id)
    test_set, test_layout = read_split(data_dir / "test_id.txt", entity2id, relation2id)

    if len(train_set) != EXPECTED_FORWARD_TRIPLES:
        raise RuntimeError(f"Parsed train split count mismatch: {len(train_set)}")
    if len(valid_set) != 6170 or len(test_set) != 6170:
        raise RuntimeError(f"Parsed valid/test counts mismatch: valid={len(valid_set)}, test={len(test_set)}")

    entity_names = {int(identifier): name for name, identifier in graph.entity2id.items()}
    relation_names = relation_display_map(graph.relation2id)
    n_forward = int(graph.num_forward_relations)

    # ---------- frozen model ----------
    model, _config, checkpoint_audit = load_frozen_rcdgate(
        nbfnet_root=args.nbfnet_root,
        rcdgate_root=args.rcdgate_root,
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        device=args.device,
    )
    model.eval()
    if len(model.layers) != EXPECTED_NUM_LAYERS:
        raise RuntimeError(f"Layer count mismatch: {len(model.layers)}")
    if int(model.path_topk) != 10:
        raise RuntimeError(f"path_topk mismatch: {model.path_topk}")
    if sum(p.numel() for p in model.parameters()) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Parameter count mismatch: {sum(p.numel() for p in model.parameters())}"
        )
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Model parameter gradients were populated before recovery")

    device = next(model.parameters()).device
    data = graph.to_pyg(device)
    state_before = model_state_sha256(model)

    # ---------- query gradient recovery ----------
    gradient_stats = {}
    attribution_by_qid = {}
    reconstruction_rows = []
    representative_edge_rows = []
    reconstruction_pass = True

    for qid in TARGET_QUERIES:
        query = resolved_by_qid[qid]
        batch = torch.tensor(
            [[query["head_id"], query["tail_id"], query["relation_id"]]],
            dtype=torch.long,
            device=device,
        )

        normal_score = float(ordinary_query_score(model, data, batch))
        with torch.enable_grad():
            attribution = attribute_query(model, data, batch)
        attribution_by_qid[qid] = attribution

        if len(attribution.edge_gradients) != EXPECTED_NUM_LAYERS:
            raise RuntimeError(
                f"{qid}: expected {EXPECTED_NUM_LAYERS} edge-gradient layers, "
                f"observed {len(attribution.edge_gradients)}"
            )
        if any(int(g.numel()) != directed_edge_count for g in attribution.edge_gradients):
            raise RuntimeError(f"{qid}: edge-gradient length mismatch")
        if not all(torch.isfinite(g).all() for g in attribution.edge_gradients):
            raise RuntimeError(f"{qid}: non-finite edge gradients")

        grads = np.stack(
            [g.detach().cpu().numpy().astype(np.float64, copy=False) for g in attribution.edge_gradients],
            axis=0,
        )
        np.savez_compressed(
            output / f"{qid}_layerwise_edge_gradients.npz",
            gradients=grads,
            directed_edge_index=np.arange(directed_edge_count, dtype=np.int64),
            source_id=np.asarray(src_cpu, dtype=np.int64),
            relation_id=np.asarray(rel_cpu, dtype=np.int64),
            target_id=np.asarray(dst_cpu, dtype=np.int64),
        )

        gradient_stats[qid] = {
            "normal_score": normal_score,
            "path_mode_score": float(attribution.score),
            "shape": list(grads.shape),
            "finite_count": int(np.isfinite(grads).sum()),
            "nan_count": int(np.isnan(grads).sum()),
            "positive_count": int((grads > 0).sum()),
            "zero_count": int((grads == 0).sum()),
            "negative_count": int((grads < 0).sum()),
            "min": float(grads.min()),
            "max": float(grads.max()),
            "mean": float(grads.mean()),
        }

        # Reconstruct all frozen Top-10 path scores from hop-specific layer gradients.
        for record in records_by_qid[qid]:
            triples = path_triples(record)
            if len(triples) > EXPECTED_NUM_LAYERS:
                raise RuntimeError(f"{qid} rank {record['path_rank']}: path longer than model depth")
            hop_values = []
            edge_indices = []
            for layer_idx, triple in enumerate(triples):
                if triple not in edge_index_by_triple:
                    raise RuntimeError(
                        f"{qid} rank {record['path_rank']}: frozen path edge absent from reasoning graph: {triple}"
                    )
                edge_idx = edge_index_by_triple[triple]
                edge_indices.append(edge_idx)
                hop_values.append(float(grads[layer_idx, edge_idx]))
            reconstructed = float(sum(hop_values) / len(hop_values))
            frozen = float(record["signed_path_score"])
            diff = abs(reconstructed - frozen)
            passed = bool(diff <= RECONSTRUCTION_TOLERANCE)
            reconstruction_pass = reconstruction_pass and passed
            reconstruction_rows.append(
                {
                    "query_id": qid,
                    "path_rank": int(record["path_rank"]),
                    "hop_count": len(triples),
                    "frozen_signed_path_score": frozen,
                    "reconstructed_signed_path_score": reconstructed,
                    "absolute_difference": diff,
                    "passed": passed,
                    "directed_edge_indices": json.dumps(edge_indices),
                    "hop_specific_signed_gradients": json.dumps(hop_values),
                }
            )

        # Representative path per-edge raw hop-specific gradient.
        record = representative_record[qid]
        triples = path_triples(record)
        for hop, triple in enumerate(triples, start=1):
            s, r, t = triple
            edge_idx = edge_index_by_triple[triple]
            layer_values = [float(grads[layer, edge_idx]) for layer in range(EXPECTED_NUM_LAYERS)]
            canonical, origin = canonical_forward(s, r, t, n_forward)
            source_name = entity_names[s]
            target_name = entity_names[t]
            relation_name = relation_names[r]
            representative_edge_rows.append(
                {
                    "query_id": qid,
                    "representative_path_rank": REPRESENTATIVE_RANK[qid],
                    "representative_path_hop": hop,
                    "reasoning_layer": hop,
                    "directed_edge_index": edge_idx,
                    "source_id": s,
                    "source_entity_cn": source_name,
                    "relation_id": r,
                    "relation_cn": relation_name,
                    "target_id": t,
                    "target_entity_cn": target_name,
                    "path_hop_signed_gradient": float(grads[hop - 1, edge_idx]),
                    "grad_layer1": layer_values[0],
                    "grad_layer2": layer_values[1],
                    "grad_layer3": layer_values[2],
                    "grad_layer4": layer_values[3],
                    "grad_layer5": layer_values[4],
                    "aggregate_signed_edge_attribution": float(sum(layer_values)),
                    "edge_origin": origin,
                    "in_train": canonical in train_set,
                    "in_valid": canonical in valid_set,
                    "in_test": canonical in test_set,
                }
            )

    write_csv(output / "path_score_reconstruction_audit.csv", reconstruction_rows)
    write_csv(output / "representative_path_edge_attribution.csv", representative_edge_rows)

    max_reconstruction_diff = max(row["absolute_difference"] for row in reconstruction_rows)
    if not reconstruction_pass:
        # Preserve recovery/audit outputs but do NOT generate paper-facing subgraphs.
        state_after = model_state_sha256(model)
        fail_payload = {
            "version": VERSION,
            "status": "STOP_PATH_SCORE_RECONSTRUCTION_FAILED",
            "checkpoint_sha256": checkpoint_sha,
            "path_method": PATH_METHOD,
            "existing_path_export_has_edge_gradient": existing_export_has_edge_gradient,
            "gradient_stats": gradient_stats,
            "max_path_score_reconstruction_difference": max_reconstruction_diff,
            "reconstruction_tolerance": RECONSTRUCTION_TOLERANCE,
            "TOP10_PATH_SCORE_RECONSTRUCTION_PASS": False,
            "MODEL_PARAMETERS_UNCHANGED": state_before == state_after,
            "MODEL_PARAMETER_GRADS_CLEAR": all(p.grad is None for p in model.parameters()),
            "READY_FOR_FIG3_LOCAL_ATTRIBUTED_SUBGRAPH": False,
        }
        write_json(output / "local_attribution_audit.json", fail_payload)
        print("TOP10_PATH_SCORE_RECONSTRUCTION_PASS=FALSE")
        print(f"MAX_PATH_SCORE_RECONSTRUCTION_DIFFERENCE={max_reconstruction_diff:.12g}")
        print("READY_FOR_FIG3_LOCAL_ATTRIBUTED_SUBGRAPH=FALSE")
        return 2

    # ---------- local subgraph construction ----------
    display_rows_all = []
    query_edge_rows = []
    local_summary = {}
    display_non_train = 0
    display_valid_test_only = 0
    display_also_valid_or_test = 0

    for qid in TARGET_QUERIES:
        attribution = attribution_by_qid[qid]
        grads = np.stack(
            [g.detach().cpu().numpy().astype(np.float64, copy=False) for g in attribution.edge_gradients],
            axis=0,
        )
        aggregate = grads.sum(axis=0)

        rep_record = representative_record[qid]
        rep_triples = path_triples(rep_record)
        rep_triple_set = set(rep_triples)
        rep_nodes = set(int(v) for v in rep_record["node_ids"])

        # All 1-hop reasoning edges incident to any representative-path node.
        local_candidate_indices = [
            i for i, (s, r, t) in enumerate(directed_triples)
            if s in rep_nodes or t in rep_nodes
        ]

        # Positive extra candidates only, deterministic rank by attribution then ids.
        extras = []
        for edge_idx in local_candidate_indices:
            triple = directed_triples[edge_idx]
            if triple in rep_triple_set:
                continue
            value = float(aggregate[edge_idx])
            if not value > 0:
                continue
            s, r, t = triple
            extras.append((edge_idx, value, s, r, t))

        extras.sort(key=lambda x: (-x[1], x[2], x[3], x[4]))
        extra_needed = max(DISPLAY_EDGE_K - len(rep_triples), 0)
        selected_extras = extras[:extra_needed]

        rows = []

        # Representative edges are first and kept in hop order.
        for hop, triple in enumerate(rep_triples, start=1):
            s, r, t = triple
            edge_idx = edge_index_by_triple[triple]
            layer_values = [float(grads[layer, edge_idx]) for layer in range(EXPECTED_NUM_LAYERS)]
            canonical, origin = canonical_forward(s, r, t, n_forward)
            source_name = entity_names[s]
            target_name = entity_names[t]
            relation_name = relation_names[r]
            src_en, src_status = entity_display(source_name)
            tgt_en, tgt_status = entity_display(target_name)
            row = {
                "query_id": qid,
                "display_edge_rank": len(rows) + 1,
                "directed_edge_index": edge_idx,
                "source_id": s,
                "source_entity_cn": source_name,
                "source_display_en": src_en,
                "source_english_label_status": src_status,
                "relation_id": r,
                "relation_cn": relation_name,
                "relation_display_en": relation_en(relation_name),
                "target_id": t,
                "target_entity_cn": target_name,
                "target_display_en": tgt_en,
                "target_english_label_status": tgt_status,
                "grad_layer1": layer_values[0],
                "grad_layer2": layer_values[1],
                "grad_layer3": layer_values[2],
                "grad_layer4": layer_values[3],
                "grad_layer5": layer_values[4],
                "aggregate_signed_edge_attribution": float(sum(layer_values)),
                "is_representative_path": True,
                "representative_path_rank": REPRESENTATIVE_RANK[qid],
                "representative_path_hop": hop,
                "path_hop_signed_gradient": float(grads[hop - 1, edge_idx]),
                "selection_reason": "REPRESENTATIVE_PATH_FORCED",
                "local_positive_attribution_rank": "",
                "edge_origin": origin,
                "in_train_graph": canonical in train_set,
                "in_valid": canonical in valid_set,
                "in_test": canonical in test_set,
            }
            rows.append(row)

        # Extra edges are appended in frozen positive-attribution order.
        for extra_rank, (edge_idx, value, s, r, t) in enumerate(selected_extras, start=1):
            layer_values = [float(grads[layer, edge_idx]) for layer in range(EXPECTED_NUM_LAYERS)]
            canonical, origin = canonical_forward(s, r, t, n_forward)
            source_name = entity_names[s]
            target_name = entity_names[t]
            relation_name = relation_names[r]
            src_en, src_status = entity_display(source_name)
            tgt_en, tgt_status = entity_display(target_name)
            row = {
                "query_id": qid,
                "display_edge_rank": len(rows) + 1,
                "directed_edge_index": edge_idx,
                "source_id": s,
                "source_entity_cn": source_name,
                "source_display_en": src_en,
                "source_english_label_status": src_status,
                "relation_id": r,
                "relation_cn": relation_name,
                "relation_display_en": relation_en(relation_name),
                "target_id": t,
                "target_entity_cn": target_name,
                "target_display_en": tgt_en,
                "target_english_label_status": tgt_status,
                "grad_layer1": layer_values[0],
                "grad_layer2": layer_values[1],
                "grad_layer3": layer_values[2],
                "grad_layer4": layer_values[3],
                "grad_layer5": layer_values[4],
                "aggregate_signed_edge_attribution": value,
                "is_representative_path": False,
                "representative_path_rank": REPRESENTATIVE_RANK[qid],
                "representative_path_hop": "",
                "path_hop_signed_gradient": "",
                "selection_reason": "TOP_POSITIVE_LOCAL_EDGE_ATTRIBUTION",
                "local_positive_attribution_rank": extra_rank,
                "edge_origin": origin,
                "in_train_graph": canonical in train_set,
                "in_valid": canonical in valid_set,
                "in_test": canonical in test_set,
            }
            rows.append(row)

        q_non_train = sum(not bool(r["in_train_graph"]) for r in rows)
        q_valid_test_only = sum(
            (not bool(r["in_train_graph"])) and (bool(r["in_valid"]) or bool(r["in_test"]))
            for r in rows
        )
        q_also_valid_test = sum(bool(r["in_valid"]) or bool(r["in_test"]) for r in rows)
        display_non_train += q_non_train
        display_valid_test_only += q_valid_test_only
        display_also_valid_or_test += q_also_valid_test

        local_summary[qid] = {
            "representative_path_rank": REPRESENTATIVE_RANK[qid],
            "representative_path_edges": len(rep_triples),
            "representative_path_nodes": len(rep_nodes),
            "local_incident_reasoning_edge_count": len(local_candidate_indices),
            "positive_extra_candidate_count": len(extras),
            "extra_selected_count": len(selected_extras),
            "final_display_edge_count": len(rows),
            "display_non_train_edge_count": q_non_train,
            "display_valid_test_only_edge_count": q_valid_test_only,
            "display_edges_also_in_valid_or_test_count": q_also_valid_test,
        }

        write_csv(output / f"Fig3_{qid}_local_attributed_subgraph.csv", rows)
        display_rows_all.extend(rows)

        # Query edge is separate, KG-unobserved, and never counted in DISPLAY_EDGE_K.
        query = resolved_by_qid[qid]
        head_name = entity_names[int(query["head_id"])]
        tail_name = entity_names[int(query["tail_id"])]
        h_en, h_status = entity_display(head_name)
        t_en, t_status = entity_display(tail_name)
        query_edge_rows.append(
            {
                "query_id": qid,
                "source_id": int(query["head_id"]),
                "source_entity_cn": head_name,
                "source_display_en": h_en,
                "source_english_label_status": h_status,
                "relation_id": int(query["relation_id"]),
                "relation_cn": relation_names[int(query["relation_id"])],
                "relation_display_en": relation_en(relation_names[int(query["relation_id"])]),
                "target_id": int(query["tail_id"]),
                "target_entity_cn": tail_name,
                "target_display_en": t_en,
                "target_english_label_status": t_status,
                "edge_role": "KG_UNOBSERVED_PREDICTED_QUERY",
                "attribution": "NA",
                "is_training_edge": False,
                "counted_in_DISPLAY_EDGE_K": False,
            }
        )

    write_csv(output / "Fig3_local_attributed_subgraphs_ALL.csv", display_rows_all)
    write_csv(output / "Fig3_KG_unobserved_query_edges.csv", query_edge_rows)

    # ---------- parameter integrity ----------
    state_after = model_state_sha256(model)
    parameters_unchanged = state_before == state_after
    parameter_grads_clear = all(p.grad is None for p in model.parameters())

    ready = bool(
        checkpoint_sha == EXPECTED_CHECKPOINT_SHA256
        and directed_edge_count == EXPECTED_DIRECTED_EDGES
        and duplicate_types == 0
        and reconstruction_pass
        and parameters_unchanged
        and parameter_grads_clear
        and display_non_train == 0
        and display_valid_test_only == 0
    )

    audit_payload = {
        "version": VERSION,
        "status": "PASS" if ready else "FAIL",
        "path_method": PATH_METHOD,
        "selection_rule": {
            "DISPLAY_EDGE_K": DISPLAY_EDGE_K,
            "representative_path_edges": "forced",
            "extra_candidate_scope": "one-hop directed train reasoning edges incident to any representative-path node",
            "extra_filter": "aggregate_signed_edge_attribution > 0",
            "aggregate_signed_edge_attribution": "sum of the same directed edge's signed gradients over layers 1..5",
            "extra_sort": [
                "aggregate_signed_edge_attribution descending",
                "source_id ascending",
                "relation_id ascending",
                "target_id ascending",
            ],
            "manual_semantic_edge_selection_used": False,
        },
        "provenance": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "frozen_path_dir": str(path_dir),
            "existing_path_export_has_edge_gradient": existing_export_has_edge_gradient,
            "model_parameter_count": sum(p.numel() for p in model.parameters()),
            "checkpoint_audit": checkpoint_audit,
        },
        "graph": {
            "train_forward_triples": len(graph.forward_triples),
            "directed_reasoning_edges": directed_edge_count,
            "unique_directed_triples": len(counts),
            "duplicate_directed_triple_types": duplicate_types,
            "num_forward_relations": n_forward,
            "split_layouts": {
                "train": train_layout,
                "valid": valid_layout,
                "test": test_layout,
            },
            "leakage_audit": leakage,
        },
        "gradient_stats": gradient_stats,
        "path_reconstruction": {
            "tolerance": RECONSTRUCTION_TOLERANCE,
            "max_absolute_difference": max_reconstruction_diff,
            "pass": reconstruction_pass,
        },
        "local_subgraphs": local_summary,
        "parameter_integrity": {
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "model_parameters_unchanged": parameters_unchanged,
            "model_parameter_grads_clear": parameter_grads_clear,
            "optimizer_created": False,
            "training_performed": False,
            "parameter_update": False,
        },
        "gates": {
            "CHECKPOINT_MATCH": checkpoint_sha == EXPECTED_CHECKPOINT_SHA256,
            "TRAIN_ONLY_REASONING_GRAPH": (
                int(leakage.get("valid_edges_in_reasoning_graph", 0)) == 0
                and int(leakage.get("test_edges_in_reasoning_graph", 0)) == 0
            ),
            "EDGE_GRADIENT_RECOVERED": True,
            "REPRESENTATIVE_PATH_EDGE_GRADIENT_RECOVERED": True,
            "TOP10_PATH_SCORE_RECONSTRUCTION_PASS": reconstruction_pass,
            "MODEL_PARAMETERS_UNCHANGED": parameters_unchanged,
            "MODEL_PARAMETER_GRADS_CLEAR": parameter_grads_clear,
            "DISPLAY_EDGE_NOT_IN_TRAIN_COUNT": display_non_train,
            "DISPLAY_VALID_TEST_ONLY_EDGE_COUNT": display_valid_test_only,
            "DISPLAY_EDGE_ALSO_IN_VALID_OR_TEST_COUNT": display_also_valid_or_test,
            "MANUAL_SEMANTIC_EDGE_SELECTION_USED": False,
            "DISPLAY_EDGE_K": DISPLAY_EDGE_K,
            "READY_FOR_FIG3_LOCAL_ATTRIBUTED_SUBGRAPH": ready,
        },
    }
    write_json(output / "local_attribution_audit.json", audit_payload)

    manifest = {
        "version": VERSION,
        "script": str(script_path),
        "script_sha256": sha256_file(script_path),
        "inputs": {str(p): sha256_file(p) for p in required_inputs if p.is_file()},
        "output_dir": str(output),
        "target_queries": list(TARGET_QUERIES),
        "representative_ranks": REPRESENTATIVE_RANK,
        "display_edge_k": DISPLAY_EDGE_K,
        "path_reconstruction_tolerance": RECONSTRUCTION_TOLERANCE,
        "training_performed": False,
        "optimizer_created": False,
        "manual_semantic_edge_selection_used": False,
    }
    write_json(output / "reproducibility_manifest.json", manifest)

    # SHA256SUMS after all normal outputs are written. Do not include SHA256SUMS itself.
    output_files = sorted(p for p in output.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    with (output / "SHA256SUMS.txt").open("w", encoding="utf-8") as f:
        for p in output_files:
            f.write(f"{sha256_file(p)}  {p.name}\n")

    # Concise console summary
    print("=" * 72)
    print("STAGE 3A-1B — READ-ONLY LOCAL EDGE ATTRIBUTION RECOVERY")
    print("=" * 72)
    print(f"CHECKPOINT_SHA256={checkpoint_sha}")
    print(f"CHECKPOINT_MATCH={str(checkpoint_sha == EXPECTED_CHECKPOINT_SHA256).upper()}")
    print(f"TRAIN_FORWARD_TRIPLES={len(graph.forward_triples)}")
    print(f"DIRECTED_REASONING_EDGES={directed_edge_count}")
    print(f"UNIQUE_DIRECTED_TRIPLES={len(counts)}")
    print(f"DUPLICATE_DIRECTED_TRIPLE_TYPES={duplicate_types}")
    print(f"MODEL_PARAMETER_COUNT={sum(p.numel() for p in model.parameters())}")
    print(f"EXISTING_PATH_EXPORT_HAS_EDGE_GRADIENT={str(existing_export_has_edge_gradient).upper()}")
    print()

    for qid in TARGET_QUERIES:
        s = gradient_stats[qid]
        q_recon = [r for r in reconstruction_rows if r["query_id"] == qid]
        q_max = max(r["absolute_difference"] for r in q_recon)
        rep = representative_record[qid]
        rep_recon = next(
            r for r in q_recon if int(r["path_rank"]) == REPRESENTATIVE_RANK[qid]
        )
        loc = local_summary[qid]
        print(f"----- {qid} -----")
        print(f"GRADIENT_SHAPE={s['shape']}")
        print(
            f"GRADIENT_COUNTS_POS_ZERO_NEG="
            f"{s['positive_count']}/{s['zero_count']}/{s['negative_count']}"
        )
        print(f"GRADIENT_NAN_COUNT={s['nan_count']}")
        print(f"TOP10_MAX_RECON_DIFF={q_max:.12g}")
        print(f"REPRESENTATIVE_PATH_RANK={REPRESENTATIVE_RANK[qid]}")
        print(f"REPRESENTATIVE_FROZEN_SCORE={float(rep['signed_path_score']):.12g}")
        print(
            f"REPRESENTATIVE_RECONSTRUCTED_SCORE="
            f"{float(rep_recon['reconstructed_signed_path_score']):.12g}"
        )
        print(f"POSITIVE_LOCAL_EXTRA_CANDIDATES={loc['positive_extra_candidate_count']}")
        print(f"EXTRA_SELECTED={loc['extra_selected_count']}")
        print(f"FINAL_DISPLAY_EDGE_COUNT={loc['final_display_edge_count']}")
        print(f"DISPLAY_NON_TRAIN_EDGE_COUNT={loc['display_non_train_edge_count']}")
        print(
            f"DISPLAY_VALID_TEST_ONLY_EDGE_COUNT="
            f"{loc['display_valid_test_only_edge_count']}"
        )
        print()

    print(f"TOP10_PATH_SCORE_RECONSTRUCTION_PASS={str(reconstruction_pass).upper()}")
    print(f"MAX_PATH_SCORE_RECONSTRUCTION_DIFFERENCE={max_reconstruction_diff:.12g}")
    print(f"MODEL_PARAMETERS_UNCHANGED={str(parameters_unchanged).upper()}")
    print(f"MODEL_PARAMETER_GRADS_CLEAR={str(parameter_grads_clear).upper()}")
    print(f"DISPLAY_EDGE_NOT_IN_TRAIN_COUNT={display_non_train}")
    print(f"DISPLAY_VALID_TEST_ONLY_EDGE_COUNT={display_valid_test_only}")
    print(f"DISPLAY_EDGE_ALSO_IN_VALID_OR_TEST_COUNT={display_also_valid_or_test}")
    print("MANUAL_SEMANTIC_EDGE_SELECTION_USED=FALSE")
    print(f"DISPLAY_EDGE_K={DISPLAY_EDGE_K}")
    print(f"READY_FOR_FIG3_LOCAL_ATTRIBUTED_SUBGRAPH={str(ready).upper()}")
    print(f"OUTPUT_DIR={output}")
    return 0 if ready else 3


if __name__ == "__main__":
    raise SystemExit(main())
