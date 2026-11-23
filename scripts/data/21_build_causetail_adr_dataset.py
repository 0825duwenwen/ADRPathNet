#!/usr/bin/env python3
"""Build a deterministic CauseTailADR KGE/name-format dataset.

The ADR candidate set is computed dynamically as the union of target-relation
tails in train.txt, valid.txt, and test.txt. Existing split membership and row
order are preserved. The script never edits the input directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")
INPUT_FILES = (
    "train.txt",
    "valid.txt",
    "test.txt",
    "entities.dict",
    "relations.dict",
    "ADRs.list",
    "all_true_triples.txt",
)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_nonempty_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError(f"Blank line found in {path}")
    return lines


def read_triples(path: Path) -> list[tuple[str, str, str]]:
    triples = []
    for line_number, line in enumerate(read_nonempty_lines(path), start=1):
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(
                f"Bad triple format in {path} at line {line_number}: {line!r}"
            )
        triples.append((parts[0], parts[1], parts[2]))
    return triples


def read_mapping(path: Path) -> list[tuple[int, str]]:
    rows = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for line_number, line in enumerate(read_nonempty_lines(path), start=1):
        parts = line.split("\t", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Bad mapping format in {path} at line {line_number}: {line!r}"
            )
        try:
            identifier = int(parts[0])
        except ValueError as error:
            raise ValueError(
                f"Non-integer ID in {path} at line {line_number}: {parts[0]!r}"
            ) from error
        name = parts[1]
        if identifier in seen_ids:
            raise ValueError(f"Duplicate ID {identifier} in {path}")
        if name in seen_names:
            raise ValueError(f"Duplicate name {name!r} in {path}")
        seen_ids.add(identifier)
        seen_names.add(name)
        rows.append((identifier, name))
    if [identifier for identifier, _ in rows] != list(range(len(rows))):
        raise ValueError(f"IDs in {path} are not ordered and continuous from 0")
    return rows


def dedupe_preserve_order(
    triples: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    seen: set[tuple[str, str, str]] = set()
    result = []
    for triple in triples:
        if triple not in seen:
            seen.add(triple)
            result.append(triple)
    return result


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def write_triples(path: Path, triples: list[tuple[str, str, str]]) -> None:
    write_lines(path, ["\t".join(triple) for triple in triples])


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def entity_type(name: str) -> str:
    return name.split("_", 1)[0] if "_" in name else "<none>"


def relation_distribution(
    triples: list[tuple[str, str, str]],
) -> dict[str, int]:
    return dict(Counter(relation for _, relation, _ in triples))


def collect_git_status() -> dict[str, object]:
    root_result = subprocess.run(
        ["git", "status", "--short", "--branch"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    nested = PROJECT_ROOT / "baselines" / "nbfnet_pyg"
    nested_result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={nested.as_posix()}",
            "status",
            "--short",
            "--branch",
        ],
        cwd=nested,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return {
        "project_root": {
            "returncode": root_result.returncode,
            "stdout": root_result.stdout.strip(),
            "stderr": root_result.stderr.strip(),
        },
        "nested_nbfnet_repository": {
            "returncode": nested_result.returncode,
            "stdout": nested_result.stdout.strip(),
            "stderr": nested_result.stderr.strip(),
        },
    }


def output_hashes(directory: Path) -> dict[str, str]:
    excluded = {"generation_manifest.json"}
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def build_dataset(
    input_dir: Path,
    output_dir: Path,
    target_relation: str,
    contains_relation: str,
    affects_relation: str,
) -> dict[str, object]:
    missing = [name for name in INPUT_FILES if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required input files: " + ", ".join(sorted(missing))
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is non-empty: {output_dir}")

    old_entities = read_mapping(input_dir / "entities.dict")
    old_relations = read_mapping(input_dir / "relations.dict")
    expected_relations = [
        (0, contains_relation),
        (1, affects_relation),
        (2, target_relation),
    ]
    if old_relations != expected_relations:
        raise ValueError(
            "relations.dict does not match the required unchanged mapping. "
            f"Expected {expected_relations!r}, got {old_relations!r}"
        )

    triples_by_split = {
        split: read_triples(input_dir / f"{split}.txt") for split in SPLITS
    }
    allowed_relations = {target_relation, contains_relation, affects_relation}
    unexpected = sorted(
        {
            relation
            for triples in triples_by_split.values()
            for _, relation, _ in triples
            if relation not in allowed_relations
        }
    )
    if unexpected:
        raise ValueError(f"Unexpected relations found: {unexpected!r}")

    cause_adrs = {
        tail
        for triples in triples_by_split.values()
        for _, relation, tail in triples
        if relation == target_relation
    }
    old_adrs = read_nonempty_lines(input_dir / "ADRs.list")
    if len(old_adrs) != len(set(old_adrs)):
        raise ValueError("Duplicate ADR names found in the input ADRs.list")
    missing_cause_adrs = sorted(cause_adrs - set(old_adrs))
    if missing_cause_adrs:
        raise ValueError(
            "Cause-tail ADRs missing from input ADRs.list: "
            + repr(missing_cause_adrs[:20])
        )
    new_adrs = [adr for adr in old_adrs if adr in cause_adrs]

    filtered_by_split: dict[str, list[tuple[str, str, str]]] = {}
    for split, triples in triples_by_split.items():
        filtered = []
        for triple in triples:
            head, relation, _ = triple
            if relation in {target_relation, contains_relation}:
                filtered.append(triple)
            elif relation == affects_relation and head in cause_adrs:
                filtered.append(triple)
        filtered_by_split[split] = filtered

    used_entities = {
        entity
        for triples in filtered_by_split.values()
        for head, _, tail in triples
        for entity in (head, tail)
    }
    old_entity_names = [name for _, name in old_entities]
    unknown_used = sorted(used_entities - set(old_entity_names))
    if unknown_used:
        raise ValueError(
            "Filtered triples contain entities absent from entities.dict: "
            + repr(unknown_used[:20])
        )
    new_entity_names = [name for name in old_entity_names if name in used_entities]
    new_entity2id = {name: index for index, name in enumerate(new_entity_names)}

    all_true = dedupe_preserve_order(
        filtered_by_split["train"]
        + filtered_by_split["valid"]
        + filtered_by_split["test"]
    )
    old_all = dedupe_preserve_order(
        triples_by_split["train"]
        + triples_by_split["valid"]
        + triples_by_split["test"]
    )
    old_entity_types = Counter(entity_type(name) for name in old_entity_names)
    new_entity_types = Counter(entity_type(name) for name in new_entity_names)
    old_relation_counts = Counter(relation for _, relation, _ in old_all)
    new_relation_counts = Counter(relation for _, relation, _ in all_true)
    removed_adrs = [adr for adr in old_adrs if adr not in cause_adrs]
    removed_entities = [
        name for name in old_entity_names if name not in used_entities
    ]
    removed_systems = [
        name for name in removed_entities if entity_type(name) == "累及系统"
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_triples(output_dir / f"{split}.txt", filtered_by_split[split])
    write_lines(
        output_dir / "entities.dict",
        [f"{index}\t{name}" for index, name in enumerate(new_entity_names)],
    )
    write_lines(
        output_dir / "relations.dict",
        [f"{identifier}\t{name}" for identifier, name in old_relations],
    )
    write_lines(output_dir / "ADRs.list", new_adrs)
    write_lines(
        output_dir / "ADR_entities.txt",
        [f"{new_entity2id[adr]}\t{adr}" for adr in new_adrs],
    )
    write_triples(output_dir / "all_true_triples.txt", all_true)

    stats = {
        "dataset_name": output_dir.name,
        "old": {
            "num_entities": len(old_entity_names),
            "entity_type_counts": dict(old_entity_types),
            "num_adrs": len(old_adrs),
            "num_triples": len(old_all),
            "split_counts": {
                split: len(triples) for split, triples in triples_by_split.items()
            },
            "relation_counts": dict(old_relation_counts),
        },
        "new": {
            "num_entities": len(new_entity_names),
            "entity_type_counts": dict(new_entity_types),
            "num_adrs": len(new_adrs),
            "num_triples": len(all_true),
            "split_counts": {
                split: len(triples)
                for split, triples in filtered_by_split.items()
            },
            "relation_counts": dict(new_relation_counts),
        },
        "removed": {
            "num_adrs": len(removed_adrs),
            "num_entities": len(removed_entities),
            "num_affects_triples": (
                old_relation_counts[affects_relation]
                - new_relation_counts[affects_relation]
            ),
            "adr_entities": removed_adrs,
            "unused_entities": removed_entities,
            "system_entities": removed_systems,
        },
        "cause_adr_count_computed": len(cause_adrs),
        "all_true_order_rule": (
            "First occurrence order in train, then valid, then test; "
            "duplicate triples after their first occurrence are omitted."
        ),
    }
    write_json(output_dir / "dataset_stats.json", stats)

    rules = f"""# {output_dir.name}: source and filtering rules

## Definition

The ADR candidate set is the union of tail entities in all `{target_relation}`
triples from `train.txt`, `valid.txt`, and `test.txt`.

## Filtering

- All `{target_relation}` triples are retained in their original split and row order.
- All `{contains_relation}` triples are retained.
- A `{affects_relation}` triple is retained only when its head belongs to the
  dynamically computed cause-tail ADR set.
- Entities unused by every retained triple are removed.
- Retained entities preserve their relative order from the source
  `entities.dict`, then receive continuous IDs from zero.
- `ADRs.list` preserves the relative order of retained ADRs in the source list.
- `ADR_entities.txt` uses `new_entity_id<TAB>ADR_entity_name` with no header.
- `all_true_triples.txt` uses first-occurrence order in train, then valid, then
  test, with later duplicates omitted.

## Provenance limitations

WHO-ART upstream files and the original package-insert extraction-to-normalized
term mappings are not present in this project. This dataset therefore applies
the directly reproducible CauseTailADR graph rule. It does not claim that every
removed entity has been proven to have been independently introduced by
WHO-ART.

The original `{input_dir.name}` dataset remains in place. Original target
relation split membership is unchanged. Results from the old and new datasets
must not be mixed as directly comparable experiment results because their
entity graph and ADR candidate spaces differ.
"""
    (output_dir / "source_and_filter_rules.md").write_text(rules, encoding="utf-8")

    manifest = {
        "dataset_name": output_dir.name,
        "generated_at": datetime.now().astimezone().isoformat(),
        "build_script": str(Path(__file__).resolve()),
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "filter_rules": {
            "target": "retain all",
            "contains": "retain all",
            "affects": "retain only when head is in computed cause-tail ADR set",
            "entity_pruning": "remove entities unused by retained triples",
            "split_membership": "unchanged",
        },
        "relations": {
            "target": target_relation,
            "contains": contains_relation,
            "affects": affects_relation,
        },
        "cause_adr_count_computed": len(cause_adrs),
        "input_sha256": {
            name: sha256_file(input_dir / name) for name in INPUT_FILES
        },
        "output_sha256": output_hashes(output_dir),
        "output_hash_exclusions": [
            "generation_manifest.json (a file cannot contain its own final hash)"
        ],
        "python_version": platform.python_version(),
        "workspace_git_status": collect_git_status(),
        "old_dataset_stats": stats["old"],
        "new_dataset_stats": stats["new"],
        "removed_adr_count": len(removed_adrs),
        "removed_affects_triple_count": stats["removed"]["num_affects_triples"],
        "removed_system_entities": removed_systems,
        "original_target_split_preserved": True,
    }
    write_json(output_dir / "generation_manifest.json", manifest)
    return stats


def refresh_manifest(
    output_dir: Path,
    nbfnet_dir: Path | None,
) -> None:
    manifest_path = output_dir / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_refreshed_at"] = datetime.now().astimezone().isoformat()
    manifest["output_sha256"] = output_hashes(output_dir)
    if nbfnet_dir is not None:
        if not nbfnet_dir.is_dir():
            raise FileNotFoundError(f"NBFNet directory not found: {nbfnet_dir}")
        manifest["nbfnet_output_dir"] = str(nbfnet_dir.resolve())
        manifest["nbfnet_output_sha256"] = output_hashes(nbfnet_dir)
    write_json(manifest_path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="data/PVKG_811")
    parser.add_argument("--output_dir", default="data/PVKG_811_CauseTailADR")
    parser.add_argument("--target_relation", default="引起")
    parser.add_argument("--contains_relation", default="包含")
    parser.add_argument("--affects_relation", default="累及")
    parser.add_argument(
        "--refresh_manifest",
        action="store_true",
        help="Refresh output hashes after validation/conversion instead of rebuilding.",
    )
    parser.add_argument(
        "--nbfnet_dir",
        default=None,
        help="Optional NBFNet directory to include when refreshing the manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = resolve_project_path(args.input_dir)
    output_dir = resolve_project_path(args.output_dir)
    nbfnet_dir = (
        resolve_project_path(args.nbfnet_dir) if args.nbfnet_dir else None
    )
    if args.refresh_manifest:
        refresh_manifest(output_dir, nbfnet_dir)
        print(f"Refreshed manifest: {output_dir / 'generation_manifest.json'}")
        return
    stats = build_dataset(
        input_dir,
        output_dir,
        args.target_relation,
        args.contains_relation,
        args.affects_relation,
    )
    print(json.dumps(stats["new"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
