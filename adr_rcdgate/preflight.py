from __future__ import annotations

import json
from pathlib import Path

from .data import TripleParseError, load_adr_ids, parse_triples, validate_fixed_counts


PREFLIGHT_FILES = (
    "entity2id.json", "relation2id.json", "train_id.txt", "valid_id.txt",
    "adr_ids.txt", "all_true_triples.txt",
)


def run_stage1_data_preflight(data_dir, output_dir):
    root = Path(data_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing preflight output: {output}")
    output.mkdir(parents=True)
    summary = {
        "version": "0.2.1",
        "status": "STOP_RCDGATE_STAGE1_DATA_PREFLIGHT",
        "train_count": None,
        "valid_count": None,
        "all_true_count": None,
        "adr_candidate_count": None,
        "train_format": None,
        "valid_format": None,
        "all_true_format": None,
        "unknown_entity_count": 0,
        "unknown_relation_count": 0,
        "test_id_loaded": False,
        "test_split_loaded": False,
        "optimizer_created": False,
        "checkpoint_written": False,
        "training_performed": False,
        "all_true_usage": "filtered ranking only",
        "all_true_used_as_training_positives": False,
        "all_true_used_as_validation_targets": False,
        "all_true_used_as_path_source": False,
        "loaded_files": list(PREFLIGHT_FILES),
        "errors": [],
    }
    try:
        paths = {name: root / name for name in PREFLIGHT_FILES}
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing required files: {missing}")
        entity2id = json.loads(paths["entity2id.json"].read_text(encoding="utf-8"))
        relation2id = json.loads(paths["relation2id.json"].read_text(encoding="utf-8"))
        train = parse_triples(paths["train_id.txt"], entity2id, relation2id)
        valid = parse_triples(paths["valid_id.txt"], entity2id, relation2id)
        all_true = parse_triples(paths["all_true_triples.txt"], entity2id, relation2id)
        adr = load_adr_ids(paths["adr_ids.txt"], entity2id)
        validate_fixed_counts(train.rows, valid.rows, all_true.rows, adr)
        summary.update({
            "train_count": len(train.rows),
            "valid_count": len(valid.rows),
            "all_true_count": len(all_true.rows),
            "adr_candidate_count": len(adr),
            "train_format": train.format,
            "valid_format": valid.format,
            "all_true_format": all_true.format,
            "status": "PASS_RCDGATE_STAGE1_DATA_PREFLIGHT",
        })
    except Exception as error:
        if isinstance(error, TripleParseError):
            if error.category == "unknown_entity":
                summary["unknown_entity_count"] = 1
            elif error.category == "unknown_relation":
                summary["unknown_relation_count"] = 1
        summary["errors"] = [f"{type(error).__name__}: {error}"]
    (output / "stage1_data_preflight_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
