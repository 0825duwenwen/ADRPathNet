from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple


EXPECTED_COUNTS = {
    "train": 65148,
    "valid": 6170,
    "all_true": 77488,
    "adr_candidates": 1460,
}


class TripleParseError(ValueError):
    """Strict parsing error carrying the unknown-token category."""

    def __init__(self, message: str, category: str = "invalid_format"):
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class ParsedTriples:
    # Internal model order is (head_id, tail_id, relation_id).
    rows: List[Tuple[int, int, int]]
    format: str


def _mapping_values(mapping: Mapping[str, int], name: str) -> Dict[int, str]:
    inverse: Dict[int, str] = {}
    for token, identifier in mapping.items():
        if isinstance(identifier, bool) or not isinstance(identifier, int):
            raise ValueError(f"{name} must map names to integer ids: {token!r} -> {identifier!r}")
        if identifier in inverse:
            raise ValueError(f"{name} contains duplicate id {identifier}")
        inverse[identifier] = str(token)
    return inverse


def parse_triples(path, entity2id, relation2id) -> ParsedTriples:
    """Parse strict head/relation/tail triples in either IDs or names.

    For each non-empty row all three fields are first parsed strictly as
    integers. If any field fails, the complete row is interpreted as names.
    This is format detection by parsing, never by language or character set.
    """
    path = Path(path)
    entity_ids = _mapping_values(entity2id, "entity2id")
    relation_ids = _mapping_values(relation2id, "relation2id")
    rows: List[Tuple[int, int, int]] = []
    formats = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_no, raw_line in enumerate(stream, start=1):
            raw = raw_line.rstrip("\r\n")
            if not raw.strip():
                continue
            if "\t" in raw:
                # A tab-delimited name may legally contain ASCII spaces. Never
                # apply whitespace splitting after detecting the delimiter.
                fields = [field.strip() for field in raw.split("\t")]
            else:
                # Compatibility for the formal integer ID files, which may be
                # separated by spaces rather than tabs.
                fields = raw.split()
            if len(fields) != 3:
                raise TripleParseError(
                    f"Invalid triple at {path}:{line_no}: expected exactly "
                    f"head relation tail, found {len(fields)} columns in raw line {raw!r}"
                )
            head_token, relation_token, tail_token = fields
            try:
                head_id, relation_id, tail_id = (int(token, 10) for token in fields)
                row_format = "integer_ids"
            except ValueError:
                row_format = "names"
                if head_token not in entity2id:
                    raise TripleParseError(
                        f"Unknown head entity token {head_token!r} at {path}:{line_no}",
                        "unknown_entity",
                    )
                if relation_token not in relation2id:
                    raise TripleParseError(
                        f"Unknown relation token {relation_token!r} at {path}:{line_no}",
                        "unknown_relation",
                    )
                if tail_token not in entity2id:
                    raise TripleParseError(
                        f"Unknown tail entity token {tail_token!r} at {path}:{line_no}",
                        "unknown_entity",
                    )
                head_id = int(entity2id[head_token])
                relation_id = int(relation2id[relation_token])
                tail_id = int(entity2id[tail_token])
            if head_id not in entity_ids:
                raise TripleParseError(
                    f"Unknown head entity id token {head_token!r} at {path}:{line_no}",
                    "unknown_entity",
                )
            if relation_id not in relation_ids:
                raise TripleParseError(
                    f"Unknown relation id token {relation_token!r} at {path}:{line_no}",
                    "unknown_relation",
                )
            if tail_id not in entity_ids:
                raise TripleParseError(
                    f"Unknown tail entity id token {tail_token!r} at {path}:{line_no}",
                    "unknown_entity",
                )
            # Preserve direction; do not add inverse triples here.
            rows.append((head_id, tail_id, relation_id))
            formats.add(row_format)
    detected = next(iter(formats)) if len(formats) == 1 else "mixed" if formats else "empty"
    return ParsedTriples(rows=rows, format=detected)


def load_adr_ids(path, entity2id):
    path = Path(path)
    valid_ids = set(_mapping_values(entity2id, "entity2id"))
    values = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, raw_line in enumerate(stream, start=1):
            token = raw_line.strip()
            if not token:
                continue
            try:
                identifier = int(token, 10)
            except ValueError:
                if token not in entity2id:
                    raise TripleParseError(
                        f"Unknown ADR entity token {token!r} at {path}:{line_no}",
                        "unknown_entity",
                    )
                identifier = int(entity2id[token])
            if identifier not in valid_ids:
                raise TripleParseError(
                    f"Unknown ADR entity id token {token!r} at {path}:{line_no}",
                    "unknown_entity",
                )
            values.append(identifier)
    return values


def validate_fixed_counts(train, valid, all_true, adr):
    observed = {
        "train": len(train), "valid": len(valid),
        "all_true": len(all_true), "adr_candidates": len(adr),
    }
    mismatch = {
        key: {"expected": EXPECTED_COUNTS[key], "observed": value}
        for key, value in observed.items() if value != EXPECTED_COUNTS[key]
    }
    if mismatch:
        raise ValueError(f"PVKG fixed-count validation failed: {mismatch}")
    return observed
