#!/usr/bin/env python3
"""Independently validate a CauseTailADR KGE dataset and optional NBFNet copy."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def raw_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8-sig").splitlines()


def triples(path: Path, failures: list[str]) -> list[tuple[str, str, str]]:
    result = []
    for line_number, line in enumerate(raw_lines(path), start=1):
        if not line.strip():
            failures.append(f"blank line: {path}:{line_number}")
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            failures.append(f"bad triple columns: {path}:{line_number}")
            continue
        result.append(tuple(parts))
    return result


def mapping(
    path: Path, failures: list[str], require_contiguous: bool = True
) -> tuple[list[tuple[int, str]], dict[str, int]]:
    result = []
    names: dict[str, int] = {}
    ids = set()
    for line_number, line in enumerate(raw_lines(path), start=1):
        if not line.strip():
            failures.append(f"blank line: {path}:{line_number}")
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            failures.append(f"bad mapping columns: {path}:{line_number}")
            continue
        try:
            identifier = int(parts[0])
        except ValueError:
            failures.append(f"bad mapping ID: {path}:{line_number}")
            continue
        if identifier in ids:
            failures.append(f"duplicate mapping ID: {path}:{line_number}")
        if parts[1] in names:
            failures.append(f"duplicate mapping name: {path}:{line_number}")
        ids.add(identifier)
        names[parts[1]] = identifier
        result.append((identifier, parts[1]))
    if require_contiguous and [
        identifier for identifier, _ in result
    ] != list(range(len(result))):
        failures.append(f"mapping IDs are not continuous and ordered: {path}")
    return result, names


def dedupe_order(items: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    result = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def add_check(
    checks: dict[str, bool],
    failures: list[str],
    name: str,
    condition: bool,
    detail: str = "",
) -> None:
    checks[name] = bool(condition)
    if not condition:
        failures.append(f"{name}: {detail}".rstrip())


def validate(
    input_dir: Path,
    output_dir: Path,
    nbfnet_dir: Path | None,
    target_relation: str,
    contains_relation: str,
    affects_relation: str,
    require_final_artifacts: bool,
) -> dict[str, object]:
    failures: list[str] = []
    checks: dict[str, bool] = {}
    core_required = {
        "train.txt",
        "valid.txt",
        "test.txt",
        "entities.dict",
        "relations.dict",
        "ADRs.list",
        "ADR_entities.txt",
        "all_true_triples.txt",
        "dataset_stats.json",
        "generation_manifest.json",
        "source_and_filter_rules.md",
    }
    if require_final_artifacts:
        core_required |= {"validation_report.json", "path_coverage_report.json"}
    missing = sorted(name for name in core_required if not (output_dir / name).is_file())
    add_check(checks, failures, "required_files_exist", not missing, repr(missing))

    source = {split: triples(input_dir / f"{split}.txt", failures) for split in SPLITS}
    new = {split: triples(output_dir / f"{split}.txt", failures) for split in SPLITS}
    entity_rows, entity2id = mapping(output_dir / "entities.dict", failures)
    relation_rows, relation2id = mapping(output_dir / "relations.dict", failures)
    old_entity_rows, _ = mapping(input_dir / "entities.dict", failures)
    all_true = triples(output_dir / "all_true_triples.txt", failures)

    old_adrs = raw_lines(input_dir / "ADRs.list")
    new_adrs = raw_lines(output_dir / "ADRs.list")
    cause_adrs = {
        tail
        for split in SPLITS
        for _, relation, tail in source[split]
        if relation == target_relation
    }
    expected_entity_order = [
        name
        for _, name in old_entity_rows
        if any(
            name in (head, tail)
            for split in SPLITS
            for head, _, tail in new[split]
        )
    ]
    expected_adrs = [adr for adr in old_adrs if adr in cause_adrs]
    new_sets = {split: set(new[split]) for split in SPLITS}

    add_check(
        checks,
        failures,
        "relations_mapping_unchanged",
        relation_rows
        == [(0, contains_relation), (1, affects_relation), (2, target_relation)],
        repr(relation_rows),
    )
    add_check(
        checks,
        failures,
        "entity_order_preserved",
        [name for _, name in entity_rows] == expected_entity_order,
    )
    add_check(
        checks,
        failures,
        "no_duplicate_triples",
        all(len(new[split]) == len(new_sets[split]) for split in SPLITS),
    )
    add_check(
        checks,
        failures,
        "splits_disjoint",
        not (
            new_sets["train"] & new_sets["valid"]
            or new_sets["train"] & new_sets["test"]
            or new_sets["valid"] & new_sets["test"]
        ),
    )
    add_check(
        checks,
        failures,
        "valid_test_target_only",
        all(
            relation == target_relation
            for split in ("valid", "test")
            for _, relation, _ in new[split]
        ),
    )
    add_check(
        checks,
        failures,
        "contains_and_affects_train_only",
        all(
            relation not in {contains_relation, affects_relation}
            for split in ("valid", "test")
            for _, relation, _ in new[split]
        ),
    )
    for split in SPLITS:
        old_target = [
            triple for triple in source[split] if triple[1] == target_relation
        ]
        new_target = [
            triple for triple in new[split] if triple[1] == target_relation
        ]
        add_check(
            checks,
            failures,
            f"{split}_target_triples_unchanged",
            old_target == new_target,
        )
    add_check(
        checks,
        failures,
        "adr_list_order_and_set_correct",
        new_adrs == expected_adrs
        and set(new_adrs) == cause_adrs
        and len(new_adrs) == len(set(new_adrs)),
    )
    new_target_tails = {
        tail
        for split in SPLITS
        for _, relation, tail in new[split]
        if relation == target_relation
    }
    add_check(
        checks,
        failures,
        "target_tails_equal_candidates",
        new_target_tails == set(new_adrs) == cause_adrs,
    )
    train_target_tails = {
        tail
        for _, relation, tail in new["train"]
        if relation == target_relation
    }
    eval_target_tails = {
        tail
        for split in ("valid", "test")
        for _, relation, tail in new[split]
        if relation == target_relation
    }
    add_check(
        checks,
        failures,
        "valid_test_targets_seen_in_train",
        eval_target_tails <= train_target_tails,
        repr(sorted(eval_target_tails - train_target_tails)[:20]),
    )
    affect_heads = {
        head
        for split in SPLITS
        for head, relation, _ in new[split]
        if relation == affects_relation
    }
    add_check(
        checks,
        failures,
        "all_affect_heads_are_candidates",
        affect_heads <= cause_adrs,
        repr(sorted(affect_heads - cause_adrs)[:20]),
    )
    add_check(
        checks,
        failures,
        "no_affect_only_adrs",
        not (affect_heads - new_target_tails),
    )
    expected_all_true = dedupe_order(new["train"] + new["valid"] + new["test"])
    add_check(
        checks,
        failures,
        "all_true_exact_union_and_order",
        all_true == expected_all_true,
    )
    used_entities = {
        entity
        for split in SPLITS
        for head, _, tail in new[split]
        for entity in (head, tail)
    }
    add_check(
        checks,
        failures,
        "entity_mapping_exactly_used_entities",
        set(entity2id) == used_entities,
        repr(sorted(set(entity2id) ^ used_entities)[:20]),
    )
    add_check(
        checks,
        failures,
        "all_relations_mapped",
        {
            relation
            for split in SPLITS
            for _, relation, _ in new[split]
        }
        <= set(relation2id),
    )
    adr_rows, _ = mapping(
        output_dir / "ADR_entities.txt", failures, require_contiguous=False
    )
    add_check(
        checks,
        failures,
        "adr_entities_matches_candidates",
        [(identifier, name) for identifier, name in adr_rows]
        == [(entity2id[name], name) for name in new_adrs],
    )
    special_cause = "不良反应_肠局部缺血"
    removed_system = "累及系统_新生儿和婴儿异常"
    add_check(
        checks,
        failures,
        "special_cause_entity_retained_without_fabrication",
        special_cause in entity2id
        and special_cause in cause_adrs
        and special_cause not in affect_heads,
    )
    add_check(
        checks,
        failures,
        "unused_neonatal_system_removed",
        removed_system not in entity2id,
    )

    if nbfnet_dir is not None:
        nbf_required = {
            "train.txt",
            "valid.txt",
            "test.txt",
            "train_id.txt",
            "valid_id.txt",
            "test_id.txt",
            "entity2id.json",
            "relation2id.json",
            "ADRs.list",
            "adr_ids.txt",
            "all_true_triples.txt",
            "dataset_stats.json",
        }
        missing_nbf = sorted(
            name for name in nbf_required if not (nbfnet_dir / name).is_file()
        )
        add_check(
            checks,
            failures,
            "nbfnet_required_files_exist",
            not missing_nbf,
            repr(missing_nbf),
        )
        add_check(
            checks,
            failures,
            "nbfnet_has_no_processed_cache",
            not (nbfnet_dir / "processed").exists(),
        )
        json_entities = json.loads(
            (nbfnet_dir / "entity2id.json").read_text(encoding="utf-8")
        )
        json_relations = json.loads(
            (nbfnet_dir / "relation2id.json").read_text(encoding="utf-8")
        )
        add_check(
            checks,
            failures,
            "nbfnet_entity_mapping_matches_kge",
            json_entities == entity2id,
        )
        add_check(
            checks,
            failures,
            "nbfnet_relation_mapping_matches_kge",
            json_relations == relation2id,
        )
        for split in SPLITS:
            add_check(
                checks,
                failures,
                f"nbfnet_{split}_names_match_kge",
                (nbfnet_dir / f"{split}.txt").read_bytes()
                == (output_dir / f"{split}.txt").read_bytes(),
            )
            id_rows = []
            for line_number, line in enumerate(
                raw_lines(nbfnet_dir / f"{split}_id.txt"), start=1
            ):
                parts = line.split("\t")
                if len(parts) != 3:
                    failures.append(
                        f"bad NBFNet ID columns: {split}_id.txt:{line_number}"
                    )
                    continue
                try:
                    id_rows.append(tuple(int(value) for value in parts))
                except ValueError:
                    failures.append(
                        f"bad NBFNet integer: {split}_id.txt:{line_number}"
                    )
            expected_ids = [
                (entity2id[head], relation2id[relation], entity2id[tail])
                for head, relation, tail in new[split]
            ]
            add_check(
                checks,
                failures,
                f"nbfnet_{split}_ids_reversible",
                id_rows == expected_ids,
            )
        adr_ids = [
            int(line) for line in raw_lines(nbfnet_dir / "adr_ids.txt")
        ]
        add_check(
            checks,
            failures,
            "nbfnet_adr_ids_exact",
            adr_ids == [entity2id[adr] for adr in new_adrs]
            and len(adr_ids) == len(set(adr_ids)),
        )

    relation_counts = Counter(
        relation
        for split in SPLITS
        for _, relation, _ in new[split]
    )
    entity_types = Counter(
        name.split("_", 1)[0] if "_" in name else "<none>"
        for name in entity2id
    )
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "stats": {
            "entities": len(entity2id),
            "entity_types": dict(entity_types),
            "adrs": len(new_adrs),
            "splits": {split: len(new[split]) for split in SPLITS},
            "relations": dict(relation_counts),
            "all_true": len(all_true),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--nbfnet_dir", default=None)
    parser.add_argument("--target_relation", default="引起")
    parser.add_argument("--contains_relation", default="包含")
    parser.add_argument("--affects_relation", default="累及")
    parser.add_argument("--require_final_artifacts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = resolve_project_path(args.input_dir)
    output_dir = resolve_project_path(args.output_dir)
    nbfnet_dir = (
        resolve_project_path(args.nbfnet_dir) if args.nbfnet_dir else None
    )
    report = validate(
        input_dir,
        output_dir,
        nbfnet_dir,
        args.target_relation,
        args.contains_relation,
        args.affects_relation,
        args.require_final_artifacts,
    )
    report_path = resolve_project_path(args.report)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
