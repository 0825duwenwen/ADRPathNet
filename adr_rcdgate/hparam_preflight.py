from __future__ import annotations

from .config import load_config
from .overlay import construct_model
from .stage1 import parameter_counts, runtime_plan


def audit_configs(nbfnet_root, config_paths):
    reports = []
    expected_counts = {3: (63940, 291), 4: (83781, 388), 5: (103622, 485)}
    for path in config_paths:
        config = load_config(path)
        model = construct_model(nbfnet_root, True, config.model)
        actual_hidden = [int(layer.output_dim) for layer in model.layers]
        total, trainable, gate = parameter_counts(model)
        expected_total, expected_gate = expected_counts[len(config.model.hidden_dims)]
        if (total, gate) != (expected_total, expected_gate):
            raise RuntimeError("parameter counts changed from the audited 3/4/5-layer contract")
        if len(model.layers) != len(config.model.hidden_dims):
            raise RuntimeError("actual layer count does not match requested configuration")
        if actual_hidden != list(config.model.hidden_dims):
            raise RuntimeError("actual hidden dimensions do not match requested configuration")
        for layer in model.layers:
            if not hasattr(layer, "dynamic_message_relation_gate"):
                raise RuntimeError("RCDGate missing from an actual model layer")
            if layer.dynamic_message_relation_gate.gate_scale != config.model.relation_gate_scale:
                raise RuntimeError("actual gate scale does not match requested configuration")
        plan = runtime_plan(config)
        reports.append({
            "config": str(path), "requested_seed": config.seed, "runtime_seed": plan["seed"],
            "requested_input_dim": config.model.input_dim, "actual_input_dim": int(model.dims[0]),
            "requested_hidden_dims": list(config.model.hidden_dims), "actual_hidden_dims": actual_hidden,
            "requested_num_layers": len(config.model.hidden_dims), "actual_num_layers": len(model.layers),
            "requested_message_func": config.model.message_func,
            "actual_message_func": [layer.message_func for layer in model.layers],
            "requested_aggregate_func": config.model.aggregate_func,
            "actual_aggregate_func": [layer.aggregate_func for layer in model.layers],
            "requested_relation_gate_scale": config.model.relation_gate_scale,
            "actual_relation_gate_scale": [layer.dynamic_message_relation_gate.gate_scale for layer in model.layers],
            "requested_epochs": config.train.epochs, "runtime_epochs": plan["epochs"],
            "requested_batch_size": config.train.batch_size, "runtime_batch_size": plan["batch_size"],
            "requested_validation_batch_size": config.validation.batch_size,
            "runtime_validation_batch_size": plan["validation_batch_size"],
            "requested_num_negative": config.train.num_negative, "runtime_num_negative": plan["num_negative"],
            "requested_learning_rate": config.optimizer.learning_rate,
            "runtime_learning_rate": plan["learning_rate"],
            "total_parameter_count": total, "trainable_parameter_count": trainable,
            "gate_parameter_count": gate,
        })
    if any(left["total_parameter_count"] >= right["total_parameter_count"] for left, right in zip(reports, reports[1:])):
        raise RuntimeError("parameter counts must strictly increase for 3/4/5 layers")
    return reports
