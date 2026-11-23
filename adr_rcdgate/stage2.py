from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch

from .data import load_adr_ids, parse_triples
from .overlay import construct_model
from .config import load_config


VERSION = "0.2.1-final"
EXPECTED_ENTITY_COUNT = 13021
EXPECTED_RELATION2ID = {"包含": 0, "累及": 1, "引起": 2}
EXPECTED_TRAIN_COUNT = 65148
EXPECTED_TEST_COUNT = 6170
EXPECTED_ALL_TRUE_COUNT = 77488
EXPECTED_ADR_COUNT = 1460
EXPECTED_PARAMETER_COUNT = 103622
EXPECTED_SEED = 42
EXPECTED_BEST_EPOCH = 7
EXPECTED_VALID_MRR = 0.6493918835350233
EXPECTED_CHECKPOINT_SHA256 = "7e99b13d90f0c0fbff42304eaa92ae4e35316034bec7847c71dcc4dc0e568e08"
EXPECTED_CONFIG_SHA256 = "1b7f2b6701b4a85a46586558627c3a05593aaf0e4dcdb905728f39f84265be6c"
ALLOWED_STAGE1_STATUSES = {
    "GO_RCDGATE_NEXT_STAGE",
    "STRONG_GO_RCDGATE_NEXT_STAGE",
}
STAGE2_DATA_FILES = (
    "entity2id.json",
    "relation2id.json",
    "train_id.txt",
    "test_id.txt",
    "adr_ids.txt",
    "all_true_triples.txt",
)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_state_dict(state_dict):
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _validate_stage1_artifacts(stage1_run_dir):
    run_dir = Path(stage1_run_dir).resolve()

    summary_path = run_dir / "stage1_summary.json"
    metrics_path = run_dir / "epoch_metrics.json"
    checkpoint_path = run_dir / "best_valid_adr_mrr.pth"
    config_path = run_dir / "config_snapshot.yaml"

    for path in (
        summary_path,
        metrics_path,
        checkpoint_path,
        config_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"missing frozen Stage 1 artifact: {path}"
            )

    summary = _json(summary_path)
    history = _json(metrics_path)
    config = load_config(config_path)

    # ----- frozen experiment identity -----
    if summary.get("version") != "0.2.1":
        raise ValueError(
            "Stage 1 summary must come from RCDGate HParam v0.2.1"
        )

    if summary.get("model") != "NBFNet-RCDGate":
        raise ValueError("model identity mismatch")

    if summary.get("seed") != EXPECTED_SEED:
        raise ValueError("seed mismatch")

    if summary.get("epochs") != 10:
        raise ValueError("Stage 1 must contain exactly 10 epochs")

    if summary.get("num_layers") != 5:
        raise ValueError("final RCDGate must contain exactly 5 layers")

    if summary.get("input_dim") != 32:
        raise ValueError("final input_dim must be 32")

    if summary.get("hidden_dims") != [32, 32, 32, 32, 32]:
        raise ValueError("final hidden_dims mismatch")

    if not math.isclose(
        float(summary.get("relation_gate_scale")),
        0.25,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("final gate scale must be 0.25")

    if not math.isclose(
        float(summary.get("learning_rate")),
        0.001,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("final learning rate must be 0.001")

    if summary.get("parameter_count") != EXPECTED_PARAMETER_COUNT:
        raise ValueError("parameter count mismatch")

    if summary.get("gate_parameter_count") != 485:
        raise ValueError("gate parameter count mismatch")

    # ----- validation selection -----
    if summary.get("best_epoch") != EXPECTED_BEST_EPOCH:
        raise ValueError("frozen best epoch mismatch")

    if not math.isclose(
        float(summary.get("best_valid_ADR_only_filtered_MRR")),
        EXPECTED_VALID_MRR,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("frozen validation MRR mismatch")

    if summary.get("checkpoint_selected_by") != \
            "validation ADR-only filtered MRR only":
        raise ValueError("checkpoint selection protocol mismatch")

    # ----- test isolation -----
    if summary.get("test_access_locked") is not True:
        raise ValueError("Stage 1 does not certify locked Test access")

    if summary.get("test_split_loaded") is not False:
        raise ValueError("Stage 1 loaded Test unexpectedly")

    if summary.get("test_evaluated") is not False:
        raise ValueError("Stage 1 evaluated Test unexpectedly")

    # ----- checkpoint path -----
    if Path(summary.get("checkpoint_path", "")).resolve() != checkpoint_path:
        raise ValueError("checkpoint path mismatch")

    # ----- frozen config identity -----
    config_sha = _sha256_file(config_path)

    if config_sha != EXPECTED_CONFIG_SHA256:
        raise ValueError(
            f"config SHA256 mismatch: {config_sha}"
        )

    if summary.get("config_sha256") != EXPECTED_CONFIG_SHA256:
        raise ValueError("summary config SHA256 mismatch")

    if config.version != "0.2.1":
        raise ValueError("config version mismatch")

    if config.seed != EXPECTED_SEED:
        raise ValueError("config seed mismatch")

    if config.model.input_dim != 32:
        raise ValueError("config input_dim mismatch")

    if list(config.model.hidden_dims) != [32, 32, 32, 32, 32]:
        raise ValueError("config hidden_dims mismatch")

    if not math.isclose(
        float(config.model.relation_gate_scale),
        0.25,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("config gate scale mismatch")

    # ----- epoch history -----
    if not isinstance(history, list) or len(history) != 10:
        raise ValueError(
            "epoch_metrics.json must contain exactly 10 epoch records"
        )

    epochs = [int(row["epoch"]) for row in history]

    if epochs != list(range(1, 11)):
        raise ValueError(
            "epoch history must cover epochs 1 through 10 exactly once"
        )

    best_row = max(
        history,
        key=lambda row: float(row["validation"]["MRR"])
    )

    if int(best_row["epoch"]) != EXPECTED_BEST_EPOCH:
        raise ValueError("epoch history best epoch mismatch")

    if not math.isclose(
        float(best_row["validation"]["MRR"]),
        EXPECTED_VALID_MRR,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("epoch history best MRR mismatch")

    # ----- checkpoint -----
    checkpoint_sha = _sha256_file(checkpoint_path)

    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"checkpoint SHA256 mismatch: {checkpoint_sha}"
        )

    payload = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(
            "checkpoint must be dictionary containing model state"
        )

    if int(payload.get("epoch", -1)) != EXPECTED_BEST_EPOCH:
        raise ValueError("checkpoint epoch mismatch")

    if not math.isclose(
        float(payload.get("valid_MRR")),
        EXPECTED_VALID_MRR,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("checkpoint Valid MRR mismatch")

    return {
        "run_dir": run_dir,
        "summary_path": summary_path,
        "metrics_path": metrics_path,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "checkpoint_payload": payload,
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": config_sha,
        "stage1_summary": summary,
        "config": config,
        "model_config": config.model,
    }

def _strict_load_frozen_model(
    nbfnet_root,
    checkpoint_payload,
    model_config,
    device,
):
    torch.manual_seed(EXPECTED_SEED)

    model = construct_model(
        nbfnet_root,
        True,
        model_config,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            "RCDGate parameter count mismatch: "
            f"expected {EXPECTED_PARAMETER_COUNT}, "
            f"observed {parameter_count}"
        )

    if len(model.layers) != 5:
        raise ValueError(
            f"expected 5 RCDGate layers, observed {len(model.layers)}"
        )

    incompatible = model.load_state_dict(
        checkpoint_payload["model"],
        strict=True,
    )

    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "strict checkpoint load failed: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, parameter_count

def run_stage2_preflight(nbfnet_root, stage1_run_dir, output_dir, device="cuda"):
    """Validate the frozen Stage 1 selection without naming or reading test data."""
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    target_device = torch.device(device)
    frozen = _validate_stage1_artifacts(stage1_run_dir)
    model, parameter_count = _strict_load_frozen_model(
        nbfnet_root,
        frozen["checkpoint_payload"],
        frozen["model_config"],
        target_device,
    )
    state_sha256 = _sha256_state_dict(model.state_dict())
    output.mkdir(parents=True)
    report = {
        "version": VERSION,
        "status": "PASS_RCDGATE_STAGE2_PREFLIGHT",
        "recommendation": "GO_RCDGATE_TEST_ONCE",
        "seed": EXPECTED_SEED,
        "best_epoch": EXPECTED_BEST_EPOCH,
        "best_valid_ADR_only_filtered_MRR": EXPECTED_VALID_MRR,
        "checkpoint_path": str(frozen["checkpoint_path"]),
        "checkpoint_sha256": frozen["checkpoint_sha256"],
        "model_state_sha256": state_sha256,
        "parameter_count": parameter_count,
        "checkpoint_load_strict": True,
        "optimizer_created": False,
        "training_performed": False,
        "test_evaluated": False,
        "test_split_loaded": False,
    }
    (output / "stage2_preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _load_test_once_data(data_dir, device):
    from torch_geometric.data import Data

    root = Path(data_dir).resolve()
    required = {name: root / name for name in STAGE2_DATA_FILES}
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required files: {missing}")

    entity2id = _json(required["entity2id.json"])
    relation2id = _json(required["relation2id.json"])
    if len(entity2id) != EXPECTED_ENTITY_COUNT:
        raise ValueError(f"expected {EXPECTED_ENTITY_COUNT} entities, observed {len(entity2id)}")
    if relation2id != EXPECTED_RELATION2ID:
        raise ValueError(f"relation mapping mismatch: {relation2id}")

    train_parsed = parse_triples(required["train_id.txt"], entity2id, relation2id)
    test_parsed = parse_triples(required["test_id.txt"], entity2id, relation2id)
    all_true_parsed = parse_triples(required["all_true_triples.txt"], entity2id, relation2id)
    adr = load_adr_ids(required["adr_ids.txt"], entity2id)
    train = train_parsed.rows
    test = test_parsed.rows
    all_true = all_true_parsed.rows

    observed = {
        "train": len(train),
        "test": len(test),
        "all_true": len(all_true),
        "adr_candidates": len(adr),
    }
    expected = {
        "train": EXPECTED_TRAIN_COUNT,
        "test": EXPECTED_TEST_COUNT,
        "all_true": EXPECTED_ALL_TRUE_COUNT,
        "adr_candidates": EXPECTED_ADR_COUNT,
    }
    if observed != expected:
        raise ValueError(f"PVKG Stage 2 fixed-count validation failed: {observed}")
    if len(set(adr)) != EXPECTED_ADR_COUNT:
        raise ValueError("ADR candidate list must contain 1460 unique entities")
    relation_id = EXPECTED_RELATION2ID["引起"]
    if any(row[2] != relation_id for row in test):
        raise ValueError("every formal test triple must use relation 引起 / id=2")
    adr_set = set(adr)
    if any(row[1] not in adr_set for row in test):
        raise ValueError("every formal test tail must belong to the 1460 ADR candidates")
    all_true_set = set(all_true)
    if any(row not in all_true_set for row in test):
        raise ValueError("all_true_triples.txt must contain every formal test triple")

    train_tensor = torch.tensor(train, dtype=torch.long, device=device)
    edge_index = train_tensor[:, [0, 1]].t()
    edge_type = train_tensor[:, 2]
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_type = torch.cat([edge_type, edge_type + len(relation2id)])
    graph = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        target_edge_index=train_tensor[:, [0, 1]].t(),
        target_edge_type=train_tensor[:, 2],
        num_nodes=len(entity2id),
    ).to(device)
    return {
        "root": root,
        "graph": graph,
        "test": test,
        "adr": adr,
        "all_true": all_true,
        "relation_id": relation_id,
        "formats": {
            "train": train_parsed.format,
            "test": test_parsed.format,
            "all_true": all_true_parsed.format,
        },
        "counts": observed,
    }


def optimistic_filtered_ranks(score_rows, triples, adr, all_true, relation_id):
    """Compute ADR-only filtered tail ranks with optimistic tie handling."""
    known = defaultdict(set)
    for head, tail, relation in all_true:
        if relation == relation_id:
            known[head].add(tail)
    position = {entity: index for index, entity in enumerate(adr)}
    ranks = []
    filtered_candidates = 0
    for head, tail, relation in triples:
        if relation != relation_id:
            continue
        if head not in score_rows:
            raise KeyError(f"missing score row for head {head}")
        if tail not in position:
            raise KeyError(f"test tail {tail} is outside the ADR candidate set")
        score = score_rows[head]
        target = score[position[tail]]
        mask = torch.ones(len(adr), dtype=torch.bool)
        for other in known[head]:
            if other != tail and other in position:
                mask[position[other]] = False
        filtered_candidates += int((~mask).sum())
        ranks.append(1 + int((score[mask] > target).sum()))
    return ranks, filtered_candidates


def _evaluate_test(model, graph, test, adr, all_true, relation_id, batch_size=16):
    selected = [row for row in test if row[2] == relation_id]
    heads = []
    seen = set()
    for head, _tail, _relation in selected:
        if head not in seen:
            seen.add(head)
            heads.append(head)

    adr_tensor = torch.tensor(adr, dtype=torch.long, device=graph.edge_index.device)
    score_rows = {}
    all_scores_finite = True
    layer_samples = [[] for _ in model.layers]
    compatibility_samples = [[] for _ in model.layers]
    for index, layer in enumerate(model.layers):
        def observe(gate, compatibility, index=index):
            layer_samples[index].append(gate.flatten()[:2048].cpu())
            compatibility_samples[index].append(compatibility.flatten()[:2048].cpu())
        layer.dynamic_message_relation_gate.statistics_observer = observe

    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(heads), batch_size):
            batch_heads = torch.tensor(
                heads[start:start + batch_size], dtype=torch.long, device=adr_tensor.device
            )
            query = torch.stack([
                batch_heads[:, None].expand(-1, len(adr)),
                adr_tensor[None, :].expand(len(batch_heads), -1),
                torch.full(
                    (len(batch_heads), len(adr)), relation_id,
                    dtype=torch.long, device=adr_tensor.device,
                ),
            ], dim=-1)
            score = model(graph, query)
            all_scores_finite = all_scores_finite and bool(torch.isfinite(score).all())
            for head, row in zip(batch_heads.tolist(), score.cpu()):
                score_rows[head] = row
    evaluation_seconds = time.perf_counter() - started
    for layer in model.layers:
        layer.dynamic_message_relation_gate.statistics_observer = None
    if not all_scores_finite:
        raise FloatingPointError("non-finite score encountered during the formal test")

    ranks, filtered_candidates = optimistic_filtered_ranks(
        score_rows, selected, adr, all_true, relation_id
    )
    if len(ranks) != EXPECTED_TEST_COUNT:
        raise ValueError(f"expected {EXPECTED_TEST_COUNT} test ranks, observed {len(ranks)}")
    rank = torch.tensor(ranks, dtype=torch.float64)
    gate_statistics = []
    for gate_parts, compatibility_parts in zip(layer_samples, compatibility_samples):
        gate_values = torch.cat(gate_parts) if gate_parts else torch.empty(0)
        compatibility_values = (
            torch.cat(compatibility_parts) if compatibility_parts else torch.empty(0)
        )
        gate_statistics.append({
            "count": int(gate_values.numel()),
            "mean": float(gate_values.mean()) if gate_values.numel() else None,
            "std": float(gate_values.std(unbiased=False)) if gate_values.numel() else None,
            "minimum": float(gate_values.min()) if gate_values.numel() else None,
            "maximum": float(gate_values.max()) if gate_values.numel() else None,
            "compatibility_term_mean": (
                float(compatibility_values.mean()) if compatibility_values.numel() else None
            ),
            "compatibility_term_std": (
                float(compatibility_values.std(unbiased=False))
                if compatibility_values.numel() else None
            ),
        })
    return {
        "MR": float(rank.mean()),
        "MRR": float((1 / rank).mean()),
        "Hits@1": float((rank <= 1).double().mean()),
        "Hits@3": float((rank <= 3).double().mean()),
        "Hits@10": float((rank <= 10).double().mean()),
        "evaluated_triples": len(ranks),
        "test_unique_heads": len(heads),
        "adr_candidate_count": len(adr),
        "filtered_candidates": filtered_candidates,
        "evaluation_seconds": evaluation_seconds,
        "all_scores_finite": all_scores_finite,
        "tie_policy": "optimistic",
        "candidate_scope": "ADR-only tail ranking",
        "layerwise_gate_value_statistics": gate_statistics,
    }


def run_test_once(
    nbfnet_root, data_dir, stage1_run_dir, preflight_report, output_dir, device="cuda"
):
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    frozen = _validate_stage1_artifacts(stage1_run_dir)
    preflight_path = Path(preflight_report).resolve()
    report = _json(preflight_path)
    if report.get("status") != "PASS_RCDGATE_STAGE2_PREFLIGHT":
        raise ValueError("Stage 2 preflight status is not PASS")
    if report.get("checkpoint_sha256") != frozen["checkpoint_sha256"]:
        raise ValueError("checkpoint changed after Stage 2 preflight")
    if Path(report.get("checkpoint_path", "")).resolve() != frozen["checkpoint_path"]:
        raise ValueError("preflight report refers to a different checkpoint")

    # Reserve the formal result directory before the first test-data read.
    output.mkdir(parents=True)
    started_record = {
        "version": VERSION,
        "status": "FORMAL_TEST_STARTED",
        "test_once": True,
        "checkpoint_path": str(frozen["checkpoint_path"]),
        "checkpoint_sha256_before": frozen["checkpoint_sha256"],
        "test_split_loaded": False,
        "test_evaluated": False,
    }
    (output / "test_once_audit.json").write_text(
        json.dumps(started_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    target_device = torch.device(device)
    data = _load_test_once_data(data_dir, target_device)
    started_record.update({
        "test_split_loaded": True,
        "data_counts": data["counts"],
        "data_formats": data["formats"],
    })
    (output / "test_once_audit.json").write_text(
        json.dumps(started_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    model, parameter_count = _strict_load_frozen_model(
        nbfnet_root,
        frozen["checkpoint_payload"],
        frozen["model_config"],
        target_device,
    )
    model_state_before = _sha256_state_dict(model.state_dict())
    if model_state_before != report.get("model_state_sha256"):
        raise ValueError("strictly loaded model state differs from the preflight state")
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    metrics = _evaluate_test(
        model, data["graph"], data["test"], data["adr"], data["all_true"],
        data["relation_id"], batch_size=16,
    )
    model_state_after = _sha256_state_dict(model.state_dict())
    checkpoint_sha256_after = _sha256_file(frozen["checkpoint_path"])
    if model_state_after != model_state_before:
        raise RuntimeError("model parameters or buffers changed during inference")
    if checkpoint_sha256_after != frozen["checkpoint_sha256"]:
        raise RuntimeError("checkpoint file changed during inference")

    result = {
        "version": VERSION,
        "status": "PASS_RCDGATE_TEST_ONCE_COMPLETE",
        "seed": EXPECTED_SEED,
        "checkpoint_selected_by": "validation ADR-only filtered MRR only",
        "checkpoint_epoch": EXPECTED_BEST_EPOCH,
        "checkpoint_valid_MRR": EXPECTED_VALID_MRR,
        "checkpoint_path": str(frozen["checkpoint_path"]),
        "checkpoint_sha256_before": frozen["checkpoint_sha256"],
        "checkpoint_sha256_after": checkpoint_sha256_after,
        "checkpoint_unchanged": True,
        "model_state_sha256_before": model_state_before,
        "model_state_sha256_after": model_state_after,
        "model_state_unchanged": True,
        "checkpoint_load_strict": True,
        "parameter_count": parameter_count,
        "optimizer_created": False,
        "training_performed": False,
        "test_split_loaded": True,
        "test_evaluated": True,
        "test_evaluation_count": 1,
        "all_true_usage": "filtered test ranking only",
        "graph_edges_from": "train_id.txt plus automatic inverse edges",
        "metrics": metrics,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(target_device))
            if target_device.type == "cuda" else 0
        ),
    }
    (output / "test_once_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audit = dict(started_record)
    audit.update({
        "status": "PASS_RCDGATE_TEST_ONCE_COMPLETE",
        "checkpoint_sha256_after": checkpoint_sha256_after,
        "checkpoint_unchanged": True,
        "model_state_sha256_before": model_state_before,
        "model_state_sha256_after": model_state_after,
        "model_state_unchanged": True,
        "checkpoint_load_strict": True,
        "parameter_count": parameter_count,
        "optimizer_created": False,
        "training_performed": False,
        "test_evaluated": True,
        "test_evaluation_count": 1,
        "result_path": str(output / "test_once_results.json"),
    })
    (output / "test_once_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
