"""Pure helpers for the CauseTailADR training and evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_DATA_FILES = (
    "train.txt",
    "valid.txt",
    "test.txt",
    "entity2id.json",
    "relation2id.json",
    "adr_ids.txt",
    "all_true_triples.txt",
)
CHECKPOINT_METADATA_VERSION = 1
TIE_POLICY = "optimistic: rank = 1 + count(candidate_score > target_score)"


def apply_static_relation_gate(message, edge_type, relation_gate):
    """Apply SGate to an explicit [batch, edge, feature] message tensor."""
    if message.ndim != 3:
        raise RuntimeError(
            "Expected message layout [batch, edge, feature], got %s"
            % (tuple(message.shape),)
        )
    if edge_type.ndim != 1 or message.shape[1] != edge_type.numel():
        raise RuntimeError(
            "edge_type must index the message edge dimension: message=%s edge_type=%s"
            % (tuple(message.shape), tuple(edge_type.shape))
        )
    edge_gate = relation_gate.index_select(0, edge_type)
    return message * edge_gate.reshape(1, edge_type.numel(), 1)


def optimistic_rank(scores, target_index):
    target_score = scores[target_index]
    return 1 + sum(score > target_score for score in scores)


def strict_load_model_state(model, state_dict):
    """Load a model with missing and unexpected keys treated as fatal."""
    return model.load_state_dict(state_dict, strict=True)


def _get(mapping, key, default=None):
    if mapping is None:
        return default
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def resolve_from_repo(path_value, repo_root):
    """Resolve a configured path without depending on the process working directory."""
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = Path(repo_root) / path
    return path.resolve()


def resolve_dataset_dir(dataset_cfg, repo_root):
    root = _get(dataset_cfg, "root")
    name = _get(dataset_cfg, "name")
    if not root or not name:
        raise ValueError("dataset.root and dataset.name are both required")
    return (resolve_from_repo(root, repo_root) / str(name)).resolve()


def normalize_gate_identity(model_cfg):
    model_class = _get(model_cfg, "class", "NBFNet")
    enabled = _get(model_cfg, "use_relation_gate", False)

    if not isinstance(enabled, bool):
        raise ValueError("model.use_relation_gate must be a boolean")

    mode = _get(model_cfg, "relation_gate_mode")


    if model_class == "NBFNetEviRerank":
        if enabled:
            raise ValueError(
                "NBFNet-EviRerank cannot be combined "
                "with a relation gate"
            )
        if mode not in (None, "none"):
            raise ValueError(
                "NBFNet-EviRerank requires "
                "relation_gate_mode to be absent "
                "or exactly 'none'"
            )

        num_slots = _get(
            model_cfg, "ingredient_num_slots", 4
        )
        slot_dim = _get(
            model_cfg, "ingredient_slot_dim", 8
        )
        slot_temperature = _get(
            model_cfg,
            "ingredient_slot_temperature",
            0.5,
        )
        reranker_hidden_dim = _get(
            model_cfg,
            "reranker_hidden_dim",
            32,
        )
        maximum_correction = _get(
            model_cfg,
            "maximum_correction",
            1.0,
        )
        contains_relation_id = _get(
            model_cfg, "contains_relation_id", 0
        )
        affects_relation_id = _get(
            model_cfg, "affects_relation_id", 1
        )
        target_relation_id = _get(
            model_cfg, "target_relation_id", 2
        )

        for value, name, minimum in (
            (num_slots, "ingredient_num_slots", 2),
            (slot_dim, "ingredient_slot_dim", 1),
            (
                reranker_hidden_dim,
                "reranker_hidden_dim",
                1,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise ValueError(
                    "NBFNet-EviRerank %s must be "
                    "an integer >= %d"
                    % (name, minimum)
                )

        if (
            isinstance(slot_temperature, bool)
            or not isinstance(
                slot_temperature, (int, float)
            )
            or slot_temperature <= 0
        ):
            raise ValueError(
                "NBFNet-EviRerank "
                "ingredient_slot_temperature must "
                "be greater than 0"
            )

        if (
            isinstance(maximum_correction, bool)
            or not isinstance(
                maximum_correction, (int, float)
            )
            or maximum_correction <= 0
        ):
            raise ValueError(
                "NBFNet-EviRerank "
                "maximum_correction must be "
                "greater than 0"
            )

        if contains_relation_id != 0:
            raise ValueError(
                "NBFNet-EviRerank requires "
                "contains_relation_id=0"
            )
        if affects_relation_id != 1:
            raise ValueError(
                "NBFNet-EviRerank requires "
                "affects_relation_id=1"
            )
        if target_relation_id != 2:
            raise ValueError(
                "NBFNet-EviRerank requires "
                "target_relation_id=2"
            )

        return {
            "model_class": "NBFNetEviRerank",
            "display_name": "NBFNet-EviRerank",
            "gate_mode": "none",
            "gate_scale": None,
            "prototype_mode": None,
            "prototype_dim": None,
            "prototype_scale_initial": None,
            "prototype_scale_max": None,
            "leave_one_drug_out": None,
            "evidence_mode":
                "set_invariant_bounded_reranking",
            "evidence_loss_mode":
                "pairwise_logistic",
            "candidate_set_invariant": True,
            "zero_initialization": True,
            "epoch_zero_protection": True,
            "ingredient_expert_mode":
                "idf_multislot_leave_one_out",
            "organ_expert_mode":
                "idf_overlap_leave_one_out",
            "ingredient_num_slots": int(
                num_slots
            ),
            "ingredient_slot_dim": int(
                slot_dim
            ),
            "ingredient_slot_temperature": float(
                slot_temperature
            ),
            "evidence_router_hidden_dim": None,
            "evidence_base_prior": None,
            "reranker_hidden_dim": int(
                reranker_hidden_dim
            ),
            "maximum_correction": float(
                maximum_correction
            ),
            "contains_relation_id": int(
                contains_relation_id
            ),
            "affects_relation_id": int(
                affects_relation_id
            ),
            "target_relation_id": int(
                target_relation_id
            ),
        }

    if model_class == "NBFNetEviMoE":
        if enabled:
            raise ValueError(
                "NBFNet-EviMoE cannot be combined "
                "with a relation gate"
            )
        if mode not in (None, "none"):
            raise ValueError(
                "NBFNet-EviMoE requires relation_gate_mode "
                "to be absent or exactly 'none'"
            )

        num_slots = _get(
            model_cfg, "ingredient_num_slots", 4
        )
        slot_dim = _get(
            model_cfg, "ingredient_slot_dim", 8
        )
        slot_temperature = _get(
            model_cfg,
            "ingredient_slot_temperature",
            0.5,
        )
        router_hidden_dim = _get(
            model_cfg,
            "evidence_router_hidden_dim",
            16,
        )
        base_prior = _get(
            model_cfg, "evidence_base_prior", 0.85
        )
        contains_relation_id = _get(
            model_cfg, "contains_relation_id", 0
        )
        affects_relation_id = _get(
            model_cfg, "affects_relation_id", 1
        )
        target_relation_id = _get(
            model_cfg, "target_relation_id", 2
        )

        for value, name, minimum in (
            (num_slots, "ingredient_num_slots", 2),
            (slot_dim, "ingredient_slot_dim", 1),
            (
                router_hidden_dim,
                "evidence_router_hidden_dim",
                1,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise ValueError(
                    "NBFNet-EviMoE %s must be an "
                    "integer >= %d" % (name, minimum)
                )

        if (
            isinstance(slot_temperature, bool)
            or not isinstance(
                slot_temperature, (int, float)
            )
            or slot_temperature <= 0
        ):
            raise ValueError(
                "NBFNet-EviMoE "
                "ingredient_slot_temperature must "
                "be greater than 0"
            )

        if (
            isinstance(base_prior, bool)
            or not isinstance(
                base_prior, (int, float)
            )
            or not 0.0 < base_prior < 1.0
        ):
            raise ValueError(
                "NBFNet-EviMoE evidence_base_prior "
                "must be strictly between 0 and 1"
            )

        if contains_relation_id != 0:
            raise ValueError(
                "NBFNet-EviMoE requires "
                "contains_relation_id=0"
            )
        if affects_relation_id != 1:
            raise ValueError(
                "NBFNet-EviMoE requires "
                "affects_relation_id=1"
            )
        if target_relation_id != 2:
            raise ValueError(
                "NBFNet-EviMoE requires "
                "target_relation_id=2"
            )

        return {
            "model_class": "NBFNetEviMoE",
            "display_name": "NBFNet-EviMoE",
            "gate_mode": "none",
            "gate_scale": None,
            "prototype_mode": None,
            "prototype_dim": None,
            "prototype_scale_initial": None,
            "prototype_scale_max": None,
            "leave_one_drug_out": None,
            "evidence_mode":
                "candidate_probability_mixture",
            "ingredient_expert_mode":
                "idf_multislot_leave_one_out",
            "organ_expert_mode":
                "idf_overlap_leave_one_out",
            "ingredient_num_slots": int(
                num_slots
            ),
            "ingredient_slot_dim": int(
                slot_dim
            ),
            "ingredient_slot_temperature": float(
                slot_temperature
            ),
            "evidence_router_hidden_dim": int(
                router_hidden_dim
            ),
            "evidence_base_prior": float(
                base_prior
            ),
            "contains_relation_id": int(
                contains_relation_id
            ),
            "affects_relation_id": int(
                affects_relation_id
            ),
            "target_relation_id": int(
                target_relation_id
            ),
        }

    if model_class == "NBFNetSIPR":
        if enabled:
            raise ValueError(
                "NBFNet-SIPR cannot be combined with a relation gate"
            )
        if mode not in (None, "none"):
            raise ValueError(
                "NBFNet-SIPR requires relation_gate_mode to be absent "
                "or exactly 'none'"
            )

        prototype_dim = _get(model_cfg, "prototype_dim", 8)
        prototype_scale = _get(model_cfg, "prototype_scale", 0.1)
        prototype_scale_max = _get(
            model_cfg, "prototype_scale_max", 1.0
        )
        contains_relation_id = _get(
            model_cfg, "contains_relation_id", 0
        )
        target_relation_id = _get(
            model_cfg, "target_relation_id", 2
        )

        if (
            isinstance(prototype_dim, bool)
            or not isinstance(prototype_dim, int)
            or prototype_dim < 1
        ):
            raise ValueError(
                "NBFNet-SIPR prototype_dim must be a positive integer"
            )
        if (
            isinstance(prototype_scale, bool)
            or not isinstance(prototype_scale, (int, float))
            or prototype_scale <= 0
        ):
            raise ValueError(
                "NBFNet-SIPR prototype_scale must be greater than 0"
            )
        if (
            isinstance(prototype_scale_max, bool)
            or not isinstance(
                prototype_scale_max, (int, float)
            )
            or prototype_scale_max <= 0
        ):
            raise ValueError(
                "NBFNet-SIPR prototype_scale_max must be "
                "greater than 0"
            )
        if prototype_scale >= prototype_scale_max:
            raise ValueError(
                "NBFNet-SIPR prototype_scale must be smaller "
                "than prototype_scale_max"
            )
        if contains_relation_id != 0:
            raise ValueError(
                "NBFNet-SIPR requires contains_relation_id=0"
            )
        if target_relation_id != 2:
            raise ValueError(
                "NBFNet-SIPR requires target_relation_id=2"
            )

        return {
            "model_class": "NBFNetSIPR",
            "gate_mode": "none",
            "gate_scale": None,
            "display_name": "NBFNet-SIPR",
            "prototype_mode":
                "sparse_ingredient_bounded",
            "prototype_dim": int(prototype_dim),
            "prototype_scale_initial": float(prototype_scale),
            "prototype_scale_max": float(
                prototype_scale_max
            ),
            "contains_relation_id": int(contains_relation_id),
            "target_relation_id": int(target_relation_id),
            "leave_one_drug_out": True,
        }

    if model_class != "NBFNet":
        raise ValueError("Unknown model class %r" % model_class)

    common = {
        "model_class": "NBFNet",
        "prototype_mode": None,
        "prototype_dim": None,
        "prototype_scale_initial": None,
        "prototype_scale_max": None,
        "contains_relation_id": None,
        "target_relation_id": None,
        "leave_one_drug_out": None,
    }

    if not enabled:
        if mode not in (None, "none"):
            raise ValueError(
                "relation_gate_mode must be absent or exactly 'none' "
                "when use_relation_gate=False"
            )
        return {
            **common,
            "gate_mode": "none",
            "gate_scale": None,
            "display_name": "NBFNet",
        }

    scale = _get(model_cfg, "relation_gate_scale")
    if scale is None or scale <= 0:
        raise ValueError(
            "relation_gate_scale must be greater than 0 "
            "when the gate is enabled"
        )

    if mode == "static":
        return {
            **common,
            "gate_mode": "static",
            "gate_scale": float(scale),
            "display_name": "NBFNet-SGate (legacy)",
        }

    if mode in ("dynamic", "query_dynamic"):
        return {
            **common,
            "gate_mode": "query_dynamic",
            "gate_scale": float(scale),
            "display_name": "NBFNet-QGate (legacy)",
        }

    if mode == "message_dynamic":
        if scale != 0.5:
            raise ValueError(
                "NBFNet-DGate requires relation_gate_scale=0.5"
            )
        return {
            **common,
            "gate_mode": "message_dynamic",
            "gate_scale": 0.5,
            "display_name": "NBFNet-DGate",
        }

    raise ValueError("Unknown relation_gate_mode %r" % mode)

def _load_name_to_id(path):
    with path.open("r", encoding="utf-8-sig") as stream:
        raw = json.load(stream)
    result = {}
    for key, value in raw.items():
        if isinstance(value, int):
            result[str(key)] = int(value)
        elif isinstance(key, str) and key.isdigit():
            result[str(value)] = int(key)
        else:
            raise ValueError("Unsupported mapping entry in %s: %r=%r" % (path, key, value))
    return result


def _read_candidate_ids(path, entity2id):
    result = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            token = line.strip()
            if not token:
                continue
            if token.isdigit():
                entity_id = int(token)
            elif token in entity2id:
                entity_id = entity2id[token]
            else:
                raise KeyError("Unknown ADR candidate %r at %s:%d" % (token, path, line_no))
            result.append(entity_id)
    if len(result) != len(set(result)):
        raise ValueError("Duplicate ADR candidate IDs in %s" % path)
    return set(result)


def _scan_triples(path, entity2id, relation2id, target_relation, adr_ids, require_target_only):
    count = 0
    relations = set()
    invalid_tails = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            text = line.strip()
            if not text:
                continue
            parts = text.split("\t")
            if len(parts) != 3:
                raise ValueError("Invalid triple at %s:%d" % (path, line_no))
            head, relation, tail = parts
            if head not in entity2id or tail not in entity2id:
                raise KeyError("Unknown entity at %s:%d" % (path, line_no))
            if relation not in relation2id:
                raise KeyError("Unknown relation %r at %s:%d" % (relation, path, line_no))
            count += 1
            relations.add(relation)
            if require_target_only and entity2id[tail] not in adr_ids:
                invalid_tails.append((line_no, tail))
    if require_target_only and relations != {target_relation}:
        raise ValueError(
            "%s must contain only target relation %r; found %r"
            % (path.name, target_relation, sorted(relations))
        )
    if invalid_tails:
        raise ValueError(
            "%s has %d tails outside the ADR candidate set (first at line %d: %r)"
            % (path.name, len(invalid_tails), invalid_tails[0][0], invalid_tails[0][1])
        )
    return count, sorted(relations)


def _triple_lines(path):
    with path.open("r", encoding="utf-8-sig") as stream:
        return {line.strip() for line in stream if line.strip()}


def preflight_dataset(dataset_dir, target_relation, expected=None):
    dataset_dir = Path(dataset_dir).resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError("Dataset directory does not exist: %s" % dataset_dir)
    missing = [name for name in REQUIRED_DATA_FILES if not (dataset_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing required dataset files under %s: %s" % (dataset_dir, missing))

    entity2id = _load_name_to_id(dataset_dir / "entity2id.json")
    relation2id = _load_name_to_id(dataset_dir / "relation2id.json")
    if target_relation not in relation2id:
        raise KeyError("Target relation %r is absent from %s" % (target_relation, dataset_dir))
    adr_ids = _read_candidate_ids(dataset_dir / "adr_ids.txt", entity2id)
    if not adr_ids:
        raise ValueError("ADR candidate set is empty")
    unknown_ids = adr_ids - set(entity2id.values())
    if unknown_ids:
        raise ValueError("ADR candidate IDs are absent from entity2id.json: %r" % sorted(unknown_ids)[:5])

    counts = {}
    split_relations = {}
    for split in ("train", "valid", "test"):
        counts[split], split_relations[split] = _scan_triples(
            dataset_dir / ("%s.txt" % split),
            entity2id,
            relation2id,
            target_relation,
            adr_ids,
            require_target_only=split in ("valid", "test"),
        )
    all_true_count, _ = _scan_triples(
        dataset_dir / "all_true_triples.txt",
        entity2id,
        relation2id,
        target_relation,
        adr_ids,
        require_target_only=False,
    )
    split_truth = set().union(
        *(_triple_lines(dataset_dir / ("%s.txt" % split)) for split in ("train", "valid", "test"))
    )
    all_true_truth = _triple_lines(dataset_dir / "all_true_triples.txt")
    if all_true_truth != split_truth:
        raise ValueError(
            "all_true_triples.txt is not exactly the train+valid+test truth set "
            "(missing=%d, extra=%d)"
            % (len(split_truth - all_true_truth), len(all_true_truth - split_truth))
        )
    report = {
        "dataset_dir": str(dataset_dir),
        "processed_path": str(dataset_dir / "processed" / "data.pt"),
        "processed_exists": (dataset_dir / "processed" / "data.pt").is_file(),
        "num_entities": len(entity2id),
        "num_forward_relations": len(relation2id),
        "num_model_relations": len(relation2id) * 2,
        "num_adr_candidates": len(adr_ids),
        "target_relation": target_relation,
        "target_relation_id": relation2id[target_relation],
        "split_counts": counts,
        "all_true_triples_count": all_true_count,
        "split_relations": split_relations,
        "paths": {name: str(dataset_dir / name) for name in REQUIRED_DATA_FILES},
    }
    expected = expected or {}
    checks = {
        "num_entities": report["num_entities"],
        "num_forward_relations": report["num_forward_relations"],
        "num_adr_candidates": report["num_adr_candidates"],
        "train": counts["train"],
        "valid": counts["valid"],
        "test": counts["test"],
        "target_relation_id": report["target_relation_id"],
    }
    mismatches = {
        key: {"expected": value, "actual": checks.get(key)}
        for key, value in expected.items()
        if key in checks and checks.get(key) != value
    }
    if mismatches:
        raise ValueError("Dataset preflight expected-count mismatch: %s" % mismatches)
    report["expected_checks"] = "passed" if expected else "not_configured"
    return report


def checkpoint_metadata(
    gate_identity, dataset_name, data_report, config_file, parameter_count=None
):
    return {
        "metadata_version": CHECKPOINT_METADATA_VERSION,
        "model_class": gate_identity["model_class"],
        "model_display_name": gate_identity["display_name"],
        "gate_mode": gate_identity["gate_mode"],
        "gate_scale": gate_identity["gate_scale"],
        "prototype_mode": gate_identity["prototype_mode"],
        "prototype_dim": gate_identity["prototype_dim"],
        "prototype_scale_initial": gate_identity[
            "prototype_scale_initial"
        ],
        "prototype_scale_max": gate_identity[
            "prototype_scale_max"
        ],
        "contains_relation_id": gate_identity[
            "contains_relation_id"
        ],
        "target_relation_id": gate_identity[
            "target_relation_id"
        ],
        "leave_one_drug_out": gate_identity[
            "leave_one_drug_out"
        ],
        "evidence_mode": gate_identity.get(
            "evidence_mode"
        ),
        "ingredient_expert_mode":
            gate_identity.get(
                "ingredient_expert_mode"
            ),
        "organ_expert_mode": gate_identity.get(
            "organ_expert_mode"
        ),
        "ingredient_num_slots": gate_identity.get(
            "ingredient_num_slots"
        ),
        "ingredient_slot_dim": gate_identity.get(
            "ingredient_slot_dim"
        ),
        "ingredient_slot_temperature":
            gate_identity.get(
                "ingredient_slot_temperature"
            ),
        "evidence_router_hidden_dim":
            gate_identity.get(
                "evidence_router_hidden_dim"
            ),
        "evidence_base_prior": gate_identity.get(
            "evidence_base_prior"
        ),
        "evidence_loss_mode": gate_identity.get(
            "evidence_loss_mode"
        ),
        "candidate_set_invariant":
            gate_identity.get(
                "candidate_set_invariant"
            ),
        "zero_initialization": gate_identity.get(
            "zero_initialization"
        ),
        "epoch_zero_protection":
            gate_identity.get(
                "epoch_zero_protection"
            ),
        "reranker_hidden_dim": gate_identity.get(
            "reranker_hidden_dim"
        ),
        "maximum_correction": gate_identity.get(
            "maximum_correction"
        ),
        "affects_relation_id": gate_identity.get(
            "affects_relation_id"
        ),
        "evimoe_hard_negative_sampling": None,
        "evimoe_hard_negative_fraction": None,
        "evimoe_exact_positive_hard_fraction": None,
        "evimoe_sparse_positive_hard_fraction": None,
        "backbone_initialization_mode": None,
        "backbone_checkpoint": None,
        "backbone_frozen": None,
        "backbone_source_model_class": None,
        "backbone_source_parameter_count": None,
        "backbone_source_best_epoch": None,
        "backbone_source_valid_mrr": None,
        "dataset_name": str(dataset_name),
        "num_entities": int(data_report["num_entities"]),
        "num_relations": int(data_report["num_model_relations"]),
        "num_forward_relations": int(data_report["num_forward_relations"]),
        "config_file": str(Path(config_file).resolve()),
        "parameter_count": parameter_count,
    }


def validate_checkpoint_metadata(checkpoint, expected, checkpoint_path, config_file, allow_legacy=False):
    actual = checkpoint.get("metadata") if isinstance(checkpoint, dict) else None
    if not isinstance(actual, dict):
        if allow_legacy:
            return "legacy"
        raise ValueError(
            "Checkpoint %s has no strict metadata; config %s may opt in with "
            "checkpoint.allow_legacy: true" % (checkpoint_path, config_file)
        )
    fields = (
        "model_class",
        "gate_mode",
        "gate_scale",
        "prototype_mode",
        "prototype_dim",
        "prototype_scale_initial",
        "prototype_scale_max",
        "contains_relation_id",
        "target_relation_id",
        "leave_one_drug_out",
        "evidence_mode",
        "ingredient_expert_mode",
        "organ_expert_mode",
        "ingredient_num_slots",
        "ingredient_slot_dim",
        "ingredient_slot_temperature",
        "evidence_router_hidden_dim",
        "evidence_base_prior",
        "evidence_loss_mode",
        "candidate_set_invariant",
        "zero_initialization",
        "epoch_zero_protection",
        "reranker_hidden_dim",
        "maximum_correction",
        "affects_relation_id",
        "evimoe_hard_negative_sampling",
        "evimoe_hard_negative_fraction",
        "evimoe_exact_positive_hard_fraction",
        "evimoe_sparse_positive_hard_fraction",
        "backbone_initialization_mode",
        "backbone_checkpoint",
        "backbone_frozen",
        "backbone_source_model_class",
        "backbone_source_parameter_count",
        "backbone_source_best_epoch",
        "backbone_source_valid_mrr",
        "dataset_name",
        "num_entities",
        "num_forward_relations",
        "num_relations",
        "parameter_count",
    )
    mismatches = {
        field: {"config": expected.get(field), "checkpoint": actual.get(field)}
        for field in fields
        if expected.get(field) != actual.get(field)
    }
    if mismatches:
        raise ValueError(
            "Checkpoint/config mismatch (config=%s, checkpoint=%s): %s"
            % (config_file, checkpoint_path, mismatches)
        )
    return "strict"
