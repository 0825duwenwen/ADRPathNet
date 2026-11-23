from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    type: str


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    hidden_dims: tuple[int, ...]
    num_relation: int
    message_func: str
    aggregate_func: str
    short_cut: bool
    layer_norm: bool
    dependent: bool
    relation_gate_scale: float


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    num_negative: int
    strict_negative: bool
    adr_negative_sampling: bool
    adversarial_temperature: float


@dataclass(frozen=True)
class OptimizerConfig:
    class_name: str
    learning_rate: float


@dataclass(frozen=True)
class ValidationConfig:
    batch_size: int
    candidate_scope: str
    filtered: bool
    tie_policy: str


@dataclass(frozen=True)
class HParamConfig:
    version: str
    experiment: ExperimentConfig
    seed: int
    model: ModelConfig
    train: TrainConfig
    optimizer: OptimizerConfig
    validation: ValidationConfig
    test_access: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["model"]["hidden_dims"] = list(self.model.hidden_dims)
        value["optimizer"]["class"] = value["optimizer"].pop("class_name")
        return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise ValueError(f"{name} keys invalid; missing={sorted(missing)}, extra={sorted(extra)}")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def parse_config(raw: Any) -> HParamConfig:
    root = _mapping(raw, "config")
    _exact_keys(root, {"version", "experiment", "seed", "model", "train", "optimizer", "validation", "test_access"}, "config")
    experiment = _mapping(root["experiment"], "experiment")
    model = _mapping(root["model"], "model")
    train = _mapping(root["train"], "train")
    optimizer = _mapping(root["optimizer"], "optimizer")
    validation = _mapping(root["validation"], "validation")
    _exact_keys(experiment, {"name", "type"}, "experiment")
    _exact_keys(model, {"input_dim", "hidden_dims", "num_relation", "message_func", "aggregate_func", "short_cut", "layer_norm", "dependent", "relation_gate_scale"}, "model")
    _exact_keys(train, {"epochs", "batch_size", "num_negative", "strict_negative", "adr_negative_sampling", "adversarial_temperature"}, "train")
    _exact_keys(optimizer, {"class", "learning_rate"}, "optimizer")
    _exact_keys(validation, {"batch_size", "candidate_scope", "filtered", "tie_policy"}, "validation")

    if root["version"] != "0.2.1":
        raise ValueError("version must be 0.2.1")
    if experiment["name"] != "NBFNet-RCDGate-HParam" or experiment["type"] != "validation_only_hyperparameter_search":
        raise ValueError("experiment identity is invalid")
    hidden = model["hidden_dims"]
    if not isinstance(hidden, list) or not hidden:
        raise ValueError("model.hidden_dims must be a non-empty list")
    hidden_dims = tuple(_positive_int(item, "model.hidden_dims item") for item in hidden)
    if model["message_func"] != "distmult":
        raise ValueError("model.message_func must be distmult")
    if model["aggregate_func"] != "pna":
        raise ValueError("model.aggregate_func must be pna")
    if optimizer["class"] != "Adam":
        raise ValueError("optimizer.class must be Adam")
    if validation["candidate_scope"] != "ADR-only" or validation["filtered"] is not True or validation["tie_policy"] != "optimistic":
        raise ValueError("validation must be ADR-only, filtered=true, tie_policy=optimistic")
    if root["test_access"] != "locked":
        raise ValueError("test_access must be locked")
    if train["strict_negative"] is not True or train["adr_negative_sampling"] is not True:
        raise ValueError("strict and ADR-constrained negative sampling must remain enabled")

    return HParamConfig(
        version=root["version"],
        experiment=ExperimentConfig(str(experiment["name"]), str(experiment["type"])),
        seed=_positive_int(root["seed"], "seed"),
        model=ModelConfig(
            _positive_int(model["input_dim"], "model.input_dim"), hidden_dims,
            _positive_int(model["num_relation"], "model.num_relation"),
            str(model["message_func"]), str(model["aggregate_func"]),
            _boolean(model["short_cut"], "model.short_cut"),
            _boolean(model["layer_norm"], "model.layer_norm"),
            _boolean(model["dependent"], "model.dependent"),
            _positive_number(model["relation_gate_scale"], "model.relation_gate_scale"),
        ),
        train=TrainConfig(
            _positive_int(train["epochs"], "train.epochs"),
            _positive_int(train["batch_size"], "train.batch_size"),
            _positive_int(train["num_negative"], "train.num_negative"),
            True, True,
            _positive_number(train["adversarial_temperature"], "train.adversarial_temperature"),
        ),
        optimizer=OptimizerConfig("Adam", _positive_number(optimizer["learning_rate"], "optimizer.learning_rate")),
        validation=ValidationConfig(
            _positive_int(validation["batch_size"], "validation.batch_size"),
            "ADR-only", True, "optimistic",
        ),
        test_access="locked",
    )


def load_config(path: str | Path) -> HParamConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML: {error}") from error
    return parse_config(raw)


def dump_config(config: HParamConfig) -> str:
    return yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True)
