from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import torch


EXPECTED_FORWARD_TRIPLES = 65148
EXPECTED_DIRECTED_EDGES = 130296


@dataclass
class TrainOnlyGraph:
    data_dir: Path
    train_file: Path
    entity2id: dict[str, int]
    relation2id: dict[str, int]
    forward_triples: list[tuple[int, int, int]]
    edge_index: torch.Tensor
    edge_type: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return len(self.entity2id)

    @property
    def num_forward_relations(self) -> int:
        return len(self.relation2id)

    @property
    def directed_edge_set(self) -> set[tuple[int, int, int]]:
        return {
            (int(h), int(t), int(r))
            for h, t, r in zip(
                self.edge_index[0].tolist(),
                self.edge_index[1].tolist(),
                self.edge_type.tolist(),
            )
        }

    def to_pyg(self, device):
        from torch_geometric.data import Data

        return Data(
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            num_nodes=self.num_nodes,
        )


def load_mapping(path: Path) -> dict[str, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    normalized: dict[str, int] = {}
    for key, value in raw.items():
        if isinstance(value, bool):
            raise ValueError(f"boolean mapping id is invalid in {path}")
        if isinstance(value, int):
            normalized[str(key)] = int(value)
        elif str(key).isdigit():
            normalized[str(value)] = int(key)
        else:
            raise ValueError(f"unsupported mapping format in {path}")
    if len(set(normalized.values())) != len(normalized):
        raise ValueError(f"duplicate ids in {path}")
    return normalized


def find_split_file(data_dir: Path, split: str, required: bool = True) -> Optional[Path]:
    candidates = [data_dir / f"{split}_id.txt", data_dir / f"{split}.txt"]
    for path in candidates:
        if path.is_file():
            return path
    if required:
        raise FileNotFoundError(f"no frozen {split} split found: {candidates}")
    return None


def parse_triples(
    path: Path,
    entity2id: Mapping[str, int],
    relation2id: Mapping[str, int],
) -> list[tuple[int, int, int]]:
    entity_ids = set(map(int, entity2id.values()))
    relation_ids = set(map(int, relation2id.values()))
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            raw = raw_line.rstrip("\r\n")
            if not raw.strip():
                continue
            fields = [part.strip() for part in raw.split("\t")] if "\t" in raw else raw.split()
            if len(fields) != 3:
                raise ValueError(f"expected head relation tail at {path}:{line_number}")
            head_token, relation_token, tail_token = fields
            try:
                head, relation, tail = map(int, fields)
            except ValueError:
                try:
                    head = int(entity2id[head_token])
                    relation = int(relation2id[relation_token])
                    tail = int(entity2id[tail_token])
                except KeyError as error:
                    raise KeyError(f"unknown token {error.args[0]!r} at {path}:{line_number}") from error
            if head not in entity_ids or tail not in entity_ids or relation not in relation_ids:
                raise ValueError(f"out-of-vocabulary id at {path}:{line_number}")
            rows.append((head, tail, relation))
    return rows


def build_train_only_graph(
    data_dir,
    expected_forward_triples: int = EXPECTED_FORWARD_TRIPLES,
    expected_directed_edges: int = EXPECTED_DIRECTED_EDGES,
) -> TrainOnlyGraph:
    root = Path(data_dir).resolve()
    entity2id = load_mapping(root / "entity2id.json")
    relation2id = load_mapping(root / "relation2id.json")
    relation_ids = sorted(relation2id.values())
    if relation_ids != list(range(len(relation_ids))):
        raise ValueError("forward relation ids must be contiguous from zero")
    train_file = find_split_file(root, "train", required=True)
    triples = parse_triples(train_file, entity2id, relation2id)
    if len(triples) != expected_forward_triples:
        raise ValueError(
            f"TRAIN_FORWARD_TRIPLES expected {expected_forward_triples}, observed {len(triples)}"
        )
    forward_index = torch.tensor([(h, t) for h, t, _ in triples], dtype=torch.long).t()
    forward_type = torch.tensor([r for _, _, r in triples], dtype=torch.long)
    edge_index = torch.cat([forward_index, forward_index.flip(0)], dim=1)
    edge_type = torch.cat([forward_type, forward_type + len(relation2id)], dim=0)
    if edge_index.shape[1] != expected_directed_edges:
        raise ValueError(
            f"TRAIN_GRAPH_EDGES_WITH_INVERSE expected {expected_directed_edges}, "
            f"observed {edge_index.shape[1]}"
        )
    return TrainOnlyGraph(
        data_dir=root,
        train_file=train_file,
        entity2id=entity2id,
        relation2id=relation2id,
        forward_triples=triples,
        edge_index=edge_index,
        edge_type=edge_type,
    )


def audit_split_leakage(graph: TrainOnlyGraph) -> dict:
    reasoning_edges = graph.directed_edge_set
    result = {}
    for split in ("valid", "test"):
        split_file = find_split_file(graph.data_dir, split, required=True)
        triples = parse_triples(split_file, graph.entity2id, graph.relation2id)
        overlap = sum(1 for triple in triples if triple in reasoning_edges)
        result[f"{split}_file"] = str(split_file)
        result[f"{split}_triple_count"] = len(triples)
        result[f"{split}_edges_in_reasoning_graph"] = overlap
    return result


def relation_display_map(relation2id: Mapping[str, int]) -> dict[int, str]:
    forward = {int(identifier): name for name, identifier in relation2id.items()}
    count = len(forward)
    result = dict(forward)
    for identifier, name in forward.items():
        result[identifier + count] = f"{name}^-1"
    return result
