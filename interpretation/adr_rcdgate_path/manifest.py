from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


EXPECTED_QUERY_IDS = [f"Q{index:02d}" for index in range(1, 13)]
REQUIRED_QUERY_FIELDS = {
    "query_id",
    "drug_display",
    "head_entity",
    "relation",
    "adr_display",
    "tail_entity",
    "discovery_rank",
}
REQUIRED_PREFLIGHT_GATES = {
    "READY_FOR_STAGE_3A1": True,
    "STAGE_3A0C_STATUS": "PASS",
    "CHECKPOINT_MATCH": True,
    "EDGE_WEIGHT_AUTOGRAD_INDEPENDENT": True,
    "PATH_GRADIENT_PASSES_RCDGATE": True,
}


class QueryManifestError(ValueError):
    pass


class QueryEntityNotFound(QueryManifestError):
    code = "QUERY_ENTITY_NOT_FOUND"


class PreflightGateError(RuntimeError):
    pass


def load_query_manifest(path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_query_manifest(manifest)


def validate_query_manifest(manifest: dict) -> dict:
    if not isinstance(manifest, dict):
        raise QueryManifestError("query manifest root must be an object")
    if manifest.get("manifest_version") != "0.2.0":
        raise QueryManifestError("manifest_version must be 0.2.0")
    if manifest.get("frozen") is not True:
        raise QueryManifestError("query manifest must be frozen")
    if manifest.get("selection_policy") != (
        "top-2 KG-unobserved predictions per drug, frozen before external validation"
    ):
        raise QueryManifestError("selection_policy does not match the frozen protocol")
    queries = manifest.get("queries")
    if not isinstance(queries, list) or len(queries) != 12:
        raise QueryManifestError("query manifest must contain exactly 12 queries")
    query_ids = []
    triples = []
    for index, query in enumerate(queries):
        if not isinstance(query, dict):
            raise QueryManifestError(f"query {index + 1} must be an object")
        if set(query) != REQUIRED_QUERY_FIELDS:
            missing = sorted(REQUIRED_QUERY_FIELDS - set(query))
            extra = sorted(set(query) - REQUIRED_QUERY_FIELDS)
            raise QueryManifestError(
                f"query {index + 1} field mismatch: missing={missing}, extra={extra}"
            )
        for field in REQUIRED_QUERY_FIELDS - {"discovery_rank"}:
            if not isinstance(query[field], str) or not query[field]:
                raise QueryManifestError(f"query {index + 1} has invalid {field}")
        if isinstance(query["discovery_rank"], bool) or not isinstance(
            query["discovery_rank"], int
        ) or query["discovery_rank"] <= 0:
            raise QueryManifestError(f"query {index + 1} has invalid discovery_rank")
        query_ids.append(query["query_id"])
        triples.append((query["head_entity"], query["relation"], query["tail_entity"]))
    if query_ids != EXPECTED_QUERY_IDS:
        raise QueryManifestError(
            f"query order must be frozen as {EXPECTED_QUERY_IDS}, observed {query_ids}"
        )
    if len(set(query_ids)) != len(query_ids):
        raise QueryManifestError("duplicate query_id")
    if len(set(triples)) != len(triples):
        raise QueryManifestError("duplicate query triple")
    return manifest


def resolve_queries(
    manifest: dict,
    entity2id: Mapping[str, int],
    relation2id: Mapping[str, int],
) -> list[dict]:
    validate_query_manifest(manifest)
    resolved = []
    for query in manifest["queries"]:
        missing = []
        if query["head_entity"] not in entity2id:
            missing.append({"role": "head", "canonical_name": query["head_entity"]})
        if query["tail_entity"] not in entity2id:
            missing.append({"role": "tail", "canonical_name": query["tail_entity"]})
        if query["relation"] not in relation2id:
            missing.append({"role": "relation", "canonical_name": query["relation"]})
        if missing:
            raise QueryEntityNotFound(
                f"QUERY_ENTITY_NOT_FOUND query_id={query['query_id']} missing={missing}"
            )
        item = dict(query)
        item.update(
            {
                "canonical_head_entity": query["head_entity"],
                "head_id": int(entity2id[query["head_entity"]]),
                "canonical_tail_entity": query["tail_entity"],
                "tail_id": int(entity2id[query["tail_entity"]]),
                "canonical_relation": query["relation"],
                "relation_id": int(relation2id[query["relation"]]),
                "resolution_method": "EXACT_CANONICAL_LOOKUP",
            }
        )
        resolved.append(item)
    return resolved


def validate_preflight_summary(summary: dict) -> dict:
    if not isinstance(summary, dict):
        raise PreflightGateError("preflight summary root must be an object")
    failures = {
        field: {"expected": expected, "observed": summary.get(field)}
        for field, expected in REQUIRED_PREFLIGHT_GATES.items()
        if summary.get(field) != expected
    }
    if failures:
        raise PreflightGateError(f"Stage 3A-0C preflight gate failed: {failures}")
    return summary


def load_preflight_summary(path) -> dict:
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_preflight_summary(summary)
