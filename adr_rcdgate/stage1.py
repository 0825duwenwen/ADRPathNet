from __future__ import annotations

import hashlib, json, math, os, time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F

from .config import HParamConfig, dump_config, load_config
from .data import load_adr_ids, parse_triples, validate_fixed_counts
from .overlay import construct_model

ALLOWED_DATA_FILES = (
    "entity2id.json", "relation2id.json", "train_id.txt", "valid_id.txt",
    "adr_ids.txt", "all_true_triples.txt",
)


def runtime_plan(config: HParamConfig):
    return {
        "seed": config.seed, "epochs": config.train.epochs,
        "batch_size": config.train.batch_size,
        "validation_batch_size": config.validation.batch_size,
        "num_negative": config.train.num_negative,
        "strict_negative": config.train.strict_negative,
        "adr_negative_sampling": config.train.adr_negative_sampling,
        "adversarial_temperature": config.train.adversarial_temperature,
        "learning_rate": config.optimizer.learning_rate,
    }


def create_optimizer(model, config: HParamConfig):
    if config.optimizer.class_name != "Adam":
        raise ValueError("only Adam is supported")
    return torch.optim.Adam(model.parameters(), lr=config.optimizer.learning_rate)


def configure_execution_policy(seed: int, device: str):
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(seed)
    return {
        "torch_manual_seed": True,
        "cuda_manual_seed_all": False,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "stage1_will_force_deterministic_algorithms": False,
    }


def freeze_config(output: Path, config: HParamConfig):
    snapshot = output / "config_snapshot.yaml"
    payload = dump_config(config).encode("utf-8")
    snapshot.write_bytes(payload)
    return snapshot, hashlib.sha256(payload).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_validation_only_data(data_dir, device):
    from torch_geometric.data import Data
    root = Path(data_dir).resolve()
    required = {name: root / name for name in ALLOWED_DATA_FILES}
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required files: {missing}")
    entity2id, relation2id = _json(required["entity2id.json"]), _json(required["relation2id.json"])
    train = parse_triples(required["train_id.txt"], entity2id, relation2id).rows
    valid = parse_triples(required["valid_id.txt"], entity2id, relation2id).rows
    all_true = parse_triples(required["all_true_triples.txt"], entity2id, relation2id).rows
    adr = load_adr_ids(required["adr_ids.txt"], entity2id)
    validate_fixed_counts(train, valid, all_true, adr)
    if len(relation2id) != 3:
        raise ValueError("expected exactly 3 forward relations")
    train_tensor = torch.tensor(train, dtype=torch.long, device=device)
    edge_index, edge_type = train_tensor[:, [0, 1]].t(), train_tensor[:, 2]
    graph = Data(
        edge_index=torch.cat([edge_index, edge_index.flip(0)], dim=1),
        edge_type=torch.cat([edge_type, edge_type + len(relation2id)]),
        target_edge_index=train_tensor[:, [0, 1]].t(), target_edge_type=train_tensor[:, 2],
        num_nodes=len(entity2id),
    ).to(device)
    return {"root": root, "graph": graph, "train": train_tensor, "valid": valid,
            "adr": adr, "all_true": all_true, "relation2id": relation2id}


def _target_relation(relation2id):
    if "引起" in relation2id:
        return int(relation2id["引起"])
    if 2 in set(map(int, relation2id.values())):
        return 2
    raise KeyError("target relation 引起 / id=2 is absent")


def _validation_metrics(model, graph, valid, adr, all_true, relation_id, batch_size):
    model.eval()
    selected = [row for row in valid if row[2] == relation_id]
    heads = list(dict.fromkeys(h for h, _t, _r in selected))
    known = defaultdict(set)
    for h, t, r in all_true:
        if r == relation_id: known[h].add(t)
    adr_tensor = torch.tensor(adr, dtype=torch.long, device=graph.edge_index.device)
    rows, started = {}, time.perf_counter()
    gates, compatibilities = [[] for _ in model.layers], [[] for _ in model.layers]
    for index, layer in enumerate(model.layers):
        def observe(gate, compatibility, index=index):
            gates[index].append(gate.flatten()[:2048].cpu())
            compatibilities[index].append(compatibility.flatten()[:2048].cpu())
        layer.dynamic_message_relation_gate.statistics_observer = observe
    with torch.inference_mode():
        for start in range(0, len(heads), batch_size):
            bh = torch.tensor(heads[start:start + batch_size], device=adr_tensor.device)
            query = torch.stack([bh[:, None].expand(-1, len(adr)),
                                 adr_tensor[None, :].expand(len(bh), -1),
                                 torch.full((len(bh), len(adr)), relation_id, device=adr_tensor.device)], dim=-1)
            for head, row in zip(bh.tolist(), model(graph, query).cpu()): rows[head] = row
    seconds = time.perf_counter() - started
    for layer in model.layers: layer.dynamic_message_relation_gate.statistics_observer = None
    position, ranks = {entity: i for i, entity in enumerate(adr)}, []
    for h, t, _r in selected:
        score, target, mask = rows[h], rows[h][position[t]], torch.ones(len(adr), dtype=torch.bool)
        for other in known[h]:
            if other != t and other in position: mask[position[other]] = False
        ranks.append(1 + int((score[mask] > target).sum()))
    rank = torch.tensor(ranks, dtype=torch.float64)
    diagnostics = []
    for gate_parts, comp_parts in zip(gates, compatibilities):
        gv = torch.cat(gate_parts) if gate_parts else torch.empty(0)
        cv = torch.cat(comp_parts) if comp_parts else torch.empty(0)
        diagnostics.append({"count": int(gv.numel()),
            "mean": float(gv.mean()) if gv.numel() else None,
            "std": float(gv.std(unbiased=False)) if gv.numel() else None,
            "minimum": float(gv.min()) if gv.numel() else None,
            "maximum": float(gv.max()) if gv.numel() else None,
            "compatibility_term_mean": float(cv.mean()) if cv.numel() else None,
            "compatibility_term_std": float(cv.std(unbiased=False)) if cv.numel() else None})
    return {"MR": float(rank.mean()), "MRR": float((1 / rank).mean()),
        "Hits@1": float((rank <= 1).double().mean()), "Hits@3": float((rank <= 3).double().mean()),
        "Hits@10": float((rank <= 10).double().mean()), "evaluated_triples": len(ranks),
        "validation_unique_heads": len(heads), "validation_seconds": seconds,
        "layerwise_gate_value_statistics": diagnostics}


def parameter_counts(model):
    parameters = list(model.parameters())
    gate_parameters = [p for layer in model.layers for p in layer.dynamic_message_relation_gate.parameters()]
    return (sum(p.numel() for p in parameters),
            sum(p.numel() for p in parameters if p.requires_grad),
            sum(p.numel() for p in gate_parameters))


def run_stage1(nbfnet_root, data_dir, output_dir, config, device="cuda"):
    if not isinstance(config, HParamConfig): config = load_config(config)
    output = Path(output_dir).resolve()
    if output.exists(): raise FileExistsError(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    run_started = time.perf_counter()
    snapshot, config_sha256 = freeze_config(output, config)
    execution_policy = configure_execution_policy(config.seed, device)
    target_device = torch.device(device)
    data = load_validation_only_data(data_dir, target_device)
    relation_id = _target_relation(data["relation2id"])
    model = construct_model(nbfnet_root, True, config.model).to(target_device)
    optimizer = create_optimizer(model, config)
    from nbfnet import tasks
    task_cfg = SimpleNamespace(adr_negative_sampling=config.train.adr_negative_sampling,
        adr_ids_path=str(data["root"] / "adr_ids.txt"), relation2id_path=str(data["root"] / "relation2id.json"),
        target_relation="引起", pp_adr_negative_sampling=False, evimoe_hard_negative_sampling=False)
    generator = torch.Generator().manual_seed(config.seed)
    best_mrr, best_epoch, history = -math.inf, None, []
    checkpoint = output / "best_valid_adr_mrr.pth"
    optimizer_steps = negative_sampling_calls = 0
    for epoch in range(1, config.train.epochs + 1):
        model.train()
        if target_device.type == "cuda": torch.cuda.reset_peak_memory_stats(target_device)
        started, losses = time.perf_counter(), []
        order = torch.randperm(len(data["train"]), generator=generator)
        for start in range(0, len(order), config.train.batch_size):
            positive = data["train"].index_select(0, order[start:start + config.train.batch_size].to(target_device))
            sampled = tasks.negative_sampling(data["graph"], positive, config.train.num_negative,
                strict=config.train.strict_negative, task_cfg=task_cfg)
            negative_sampling_calls += 1
            prediction = model(data["graph"], sampled)
            target = torch.zeros_like(prediction); target[:, 0] = 1
            element = F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
            weight = torch.ones_like(prediction)
            with torch.no_grad():
                weight[:, 1:] = F.softmax(prediction[:, 1:] / config.train.adversarial_temperature, dim=-1)
            loss = ((element * weight).sum(-1) / weight.sum(-1)).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step(); optimizer_steps += 1
            losses.append(float(loss.detach()))
        training_seconds = time.perf_counter() - started
        metrics = _validation_metrics(model, data["graph"], data["valid"], data["adr"],
            data["all_true"], relation_id, config.validation.batch_size)
        gate_diagnostics = []
        for layer in model.layers:
            gate = layer.dynamic_message_relation_gate
            gate_diagnostics.append({"message_weight_norm": float(gate.message_linear.weight.detach().norm()),
                "relation_weight_norm": float(gate.relation_linear.weight.detach().norm()),
                "compatibility_weight_norm": float(gate.compatibility_weight.detach().norm()),
                "bias": float(gate.message_linear.bias.detach()[0])})
        is_best = metrics["MRR"] > best_mrr
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses),
            "training_seconds": training_seconds, "validation_seconds": metrics["validation_seconds"],
            "validation": metrics, "is_best": is_best,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(target_device) if target_device.type == "cuda" else 0,
            "layerwise_gate_diagnostics": gate_diagnostics, "optimizer_steps": optimizer_steps,
            "negative_sampling_calls": negative_sampling_calls})
        if is_best:
            best_mrr, best_epoch = metrics["MRR"], epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "valid_MRR": best_mrr}, checkpoint)
        (output / "epoch_metrics.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    total, trainable, gate_count = parameter_counts(model)
    summary = {"version": config.version, "model": "NBFNet-RCDGate",
        "experiment_type": config.experiment.type, "seed": config.seed,
        "num_layers": len(model.layers), "input_dim": config.model.input_dim,
        "hidden_dims": list(config.model.hidden_dims), "message_func": config.model.message_func,
        "aggregate_func": config.model.aggregate_func, "relation_gate_scale": config.model.relation_gate_scale,
        "epochs": config.train.epochs, "batch_size": config.train.batch_size,
        "validation_batch_size": config.validation.batch_size, "num_negative": config.train.num_negative,
        "strict_negative": config.train.strict_negative, "adr_negative_sampling": config.train.adr_negative_sampling,
        "adversarial_temperature": config.train.adversarial_temperature,
        "optimizer": config.optimizer.class_name, "learning_rate": config.optimizer.learning_rate,
        "parameter_count": total, "trainable_parameter_count": trainable, "gate_parameter_count": gate_count,
        "best_epoch": best_epoch, "best_valid_ADR_only_filtered_MRR": best_mrr,
        "checkpoint_selected_by": "validation ADR-only filtered MRR only", "checkpoint_path": str(checkpoint),
        "candidate_scope": config.validation.candidate_scope, "filtered": config.validation.filtered,
        "tie_policy": config.validation.tie_policy, "all_true_usage": "filtered validation ranking only",
        "test_access_locked": True, "test_split_loaded": False, "test_evaluated": False,
        **execution_policy, "execution_policy": execution_policy,
        "config_snapshot_path": str(snapshot), "config_sha256": config_sha256,
        "elapsed_seconds": time.perf_counter() - run_started}
    (output / "stage1_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
