from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence


def contribution_sign(score: float) -> str:
    if score > 0:
        return "POSITIVE"
    if score < 0:
        return "NEGATIVE"
    return "ZERO"


def build_path_records(
    query_id: str,
    drug_name: str,
    adr_name: str,
    head_id: int,
    tail_id: int,
    paths: Sequence,
    path_scores: Sequence[float],
    entity_names: Mapping[int, str],
    relation_names: Mapping[int, str],
    train_edge_set: set[tuple[int, int, int]],
    max_hops: int,
) -> tuple[list[dict], dict]:
    if len(paths) != len(path_scores):
        raise ValueError("paths and path_scores length mismatch")
    records = []
    edge_count = 0
    not_in_train = 0
    continuity_failures = 0
    endpoint_failures = 0
    hop_failures = 0
    for rank, (path, score) in enumerate(zip(paths, path_scores), start=1):
        normalized = [(int(h), int(t), int(r)) for h, t, r in path]
        hop_count = len(normalized)
        edge_count += hop_count
        continuity = all(
            normalized[index][1] == normalized[index + 1][0]
            for index in range(max(0, hop_count - 1))
        )
        endpoint_ok = bool(normalized) and normalized[0][0] == head_id and normalized[-1][1] == tail_id
        hop_ok = 1 <= hop_count <= max_hops
        membership = [edge in train_edge_set for edge in normalized]
        not_in_train += sum(not value for value in membership)
        continuity_failures += int(not continuity)
        endpoint_failures += int(not endpoint_ok)
        hop_failures += int(not hop_ok)
        node_ids = [normalized[0][0]] + [edge[1] for edge in normalized] if normalized else []
        relation_ids = [edge[2] for edge in normalized]
        edge_records = [
            {
                "head_id": h,
                "head_name": entity_names[h],
                "tail_id": t,
                "tail_name": entity_names[t],
                "relation_id": r,
                "relation_name": relation_names[r],
                "in_train_graph": member,
            }
            for (h, t, r), member in zip(normalized, membership)
        ]
        numeric_score = float(score)
        records.append(
            {
                "query_id": query_id,
                "drug_name": drug_name,
                "adr_name": adr_name,
                "path_rank": rank,
                "hop_count": hop_count,
                "signed_path_score": numeric_score,
                "contribution_sign": contribution_sign(numeric_score),
                "node_ids": node_ids,
                "node_names": [entity_names[value] for value in node_ids],
                "relation_ids": relation_ids,
                "relation_names": [relation_names[value] for value in relation_ids],
                "edges": edge_records,
                "continuity_ok": continuity,
                "endpoints_ok": endpoint_ok,
                "max_hops_ok": hop_ok,
                "all_edges_in_train_graph": all(membership),
            }
        )
    audit = {
        "path_count": len(records),
        "path_edge_count": edge_count,
        "path_edge_not_in_train": not_in_train,
        "path_continuity_failures": continuity_failures,
        "path_endpoint_failures": endpoint_failures,
        "path_hop_limit_failures": hop_failures,
    }
    return records, audit


def write_path_json(path, records: Sequence[dict]) -> None:
    Path(path).write_text(
        json.dumps(list(records), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_path_csv(path, records: Sequence[dict]) -> None:
    fields = [
        "query_id",
        "drug_name",
        "adr_name",
        "path_rank",
        "hop_count",
        "signed_path_score",
        "contribution_sign",
        "node_ids",
        "node_names",
        "relation_ids",
        "relation_names",
        "edges",
        "continuity_ok",
        "endpoints_ok",
        "max_hops_ok",
        "all_edges_in_train_graph",
    ]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            for field in ("node_ids", "node_names", "relation_ids", "relation_names", "edges"):
                row[field] = json.dumps(row[field], ensure_ascii=False, separators=(",", ":"))
            writer.writerow(row)


def build_formal_path_records(
    query: dict,
    model_score: float,
    paths: Sequence,
    path_scores: Sequence[float],
    entity_names: Mapping[int, str],
    relation_names: Mapping[int, str],
    train_edge_set: set[tuple[int, int, int]],
    max_hops: int,
) -> tuple[list[dict], dict]:
    if len(paths) != len(path_scores):
        raise ValueError("paths and path_scores length mismatch")
    records = []
    audit = {
        "query_id": query["query_id"],
        "path_count": len(paths),
        "path_edge_count": 0,
        "path_edge_not_in_train": 0,
        "broken_path_count": 0,
        "wrong_start_count": 0,
        "wrong_end_count": 0,
        "over_depth_path_count": 0,
    }
    for rank, (path, path_score) in enumerate(zip(paths, path_scores), start=1):
        edges = [(int(h), int(t), int(r)) for h, t, r in path]
        hop_count = len(edges)
        audit["path_edge_count"] += hop_count
        continuity_ok = all(
            edges[index][1] == edges[index + 1][0]
            for index in range(max(0, hop_count - 1))
        )
        start_ok = bool(edges) and edges[0][0] == query["head_id"]
        end_ok = bool(edges) and edges[-1][1] == query["tail_id"]
        depth_ok = 1 <= hop_count <= max_hops
        membership = [edge in train_edge_set for edge in edges]
        audit["path_edge_not_in_train"] += sum(not value for value in membership)
        audit["broken_path_count"] += int(not continuity_ok)
        audit["wrong_start_count"] += int(not start_ok)
        audit["wrong_end_count"] += int(not end_ok)
        audit["over_depth_path_count"] += int(not depth_ok)
        node_ids = [edges[0][0]] + [edge[1] for edge in edges] if edges else []
        relation_ids = [edge[2] for edge in edges]
        edge_sequence = [
            {
                "head_id": head,
                "head_name": entity_names[head],
                "relation_id": relation,
                "relation_name": relation_names[relation],
                "tail_id": tail,
                "tail_name": entity_names[tail],
                "in_train_graph": present,
            }
            for (head, tail, relation), present in zip(edges, membership)
        ]
        numeric_score = float(path_score)
        records.append(
            {
                "query_id": query["query_id"],
                "drug_display": query["drug_display"],
                "head_entity": query["canonical_head_entity"],
                "head_id": query["head_id"],
                "relation": query["canonical_relation"],
                "relation_id": query["relation_id"],
                "adr_display": query["adr_display"],
                "tail_entity": query["canonical_tail_entity"],
                "tail_id": query["tail_id"],
                "discovery_rank": query["discovery_rank"],
                "model_score": float(model_score),
                "path_rank": rank,
                "hop_count": hop_count,
                "signed_path_score": numeric_score,
                "contribution_sign": contribution_sign(numeric_score),
                "node_ids": node_ids,
                "node_names": [entity_names[value] for value in node_ids],
                "relation_ids": relation_ids,
                "relation_names": [relation_names[value] for value in relation_ids],
                "edge_sequence": edge_sequence,
                "continuity_ok": continuity_ok,
                "start_ok": start_ok,
                "end_ok": end_ok,
                "max_depth_ok": depth_ok,
                "all_edges_in_train_graph": all(membership),
            }
        )
    return records, audit


def write_formal_path_csv(path, records: Sequence[dict]) -> None:
    fields = [
        "query_id", "drug_display", "head_entity", "head_id", "relation",
        "relation_id", "adr_display", "tail_entity", "tail_id", "discovery_rank",
        "model_score", "path_rank", "hop_count", "signed_path_score",
        "contribution_sign", "node_ids", "node_names", "relation_ids",
        "relation_names", "edge_sequence", "continuity_ok", "start_ok", "end_ok",
        "max_depth_ok", "all_edges_in_train_graph",
    ]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            for field in ("node_ids", "node_names", "relation_ids", "relation_names", "edge_sequence"):
                row[field] = json.dumps(row[field], ensure_ascii=False, separators=(",", ":"))
            writer.writerow(row)


def write_query_summary_csv(path, rows: Sequence[dict]) -> None:
    fields = [
        "query_id", "drug", "ADR", "discovery_rank", "model_score",
        "normal_score", "path_mode_score", "abs_score_diff", "path_count",
        "positive_path_count", "negative_path_count", "zero_path_count",
        "best_path_score",
    ]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_human_readable_paths(
    path,
    resolved_queries: Sequence[dict],
    records: Sequence[dict],
    query_summaries: Sequence[dict] = (),
) -> None:
    by_query = {query["query_id"]: [] for query in resolved_queries}
    summary_by_query = {row["query_id"]: row for row in query_summaries}
    for record in records:
        by_query[record["query_id"]].append(record)
    lines = []
    for query in resolved_queries:
        lines.extend(
            [
                "=" * 70,
                query["query_id"],
                query["canonical_head_entity"],
                f"--{query['canonical_relation']}-->",
                query["canonical_tail_entity"],
                "",
                f"Discovery rank: {query['discovery_rank']}",
                f"Model score: {summary_by_query.get(query['query_id'], {}).get('model_score', 'NOT_AVAILABLE')}",
                f"Path count: {len(by_query[query['query_id']])}",
            ]
        )
        for record in by_query[query["query_id"]]:
            lines.extend(
                [
                    "",
                    f"Path {record['path_rank']}",
                    f"score = {record['signed_path_score']:.12g}",
                    f"sign = {record['contribution_sign']}",
                    f"hops = {record['hop_count']}",
                    record["node_names"][0],
                ]
            )
            for edge in record["edge_sequence"]:
                lines.append(f"--{edge['relation_name']}-->")
                lines.append(edge["tail_name"])
    lines.append("=" * 70)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
