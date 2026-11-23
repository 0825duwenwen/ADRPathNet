from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch


PATCH_MARKER = "_adr_rcdgate_path_original_bellmanford"


@dataclass
class AttributionResult:
    score: torch.Tensor
    edge_weights: list[torch.Tensor]
    edge_gradients: list[torch.Tensor]
    paths: Sequence
    path_scores: Sequence[float]
    edge_weight_audit: dict
    gate_gradient_reachable: bool
    gate_gradient_finite: bool


def make_independent_edge_weights(
    base_edge_weight: torch.Tensor,
    layer_count: int,
) -> list[torch.Tensor]:
    """Create independent leaf perturbations from the same immutable base value."""
    if base_edge_weight.requires_grad:
        base_edge_weight = base_edge_weight.detach()
    values = []
    for _ in range(int(layer_count)):
        value = base_edge_weight.detach().clone().requires_grad_(True)
        values.append(value)
    return values


def audit_edge_weight_independence(edge_weights: Sequence[torch.Tensor]) -> dict:
    pointers = [int(value.data_ptr()) for value in edge_weights]
    versions_before = [int(value._version) for value in edge_weights]
    value_snapshots = [value.detach().clone() for value in edge_weights]
    mutation_isolated = True
    if edge_weights:
        with torch.no_grad():
            edge_weights[0].view(-1)[0].add_(1)
        mutation_isolated = all(
            torch.equal(value.detach(), snapshot)
            for value, snapshot in zip(edge_weights[1:], value_snapshots[1:])
        )
        with torch.no_grad():
            edge_weights[0].copy_(value_snapshots[0])
    return {
        "layer_count": len(edge_weights),
        "all_requires_grad": all(value.requires_grad for value in edge_weights),
        "all_leaf": all(value.is_leaf for value in edge_weights),
        "all_grad_fn_none": all(value.grad_fn is None for value in edge_weights),
        "all_storage_pointers_distinct": len(pointers) == len(set(pointers)),
        "mutation_isolated": mutation_isolated,
        "versions_before": versions_before,
        "autograd_independent": all(value.is_leaf and value.grad_fn is None for value in edge_weights)
        and len(pointers) == len(set(pointers))
        and mutation_isolated,
    }


def _run_independent_bellmanford(
    model,
    data,
    h_index: torch.Tensor,
    r_index: torch.Tensor,
    edge_weight_values: Optional[Sequence[torch.Tensor]] = None,
    requires_grad: bool = True,
):
    """The audited Bellman-Ford body with only path edge-weight creation changed."""
    batch_size = len(r_index)
    query = model.query(r_index)
    index = h_index.unsqueeze(-1).expand_as(query)
    boundary = torch.zeros(
        batch_size,
        data.num_nodes,
        model.dims[0],
        device=h_index.device,
        dtype=query.dtype,
    )
    boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
    size = (data.num_nodes, data.num_nodes)
    base_edge_weight = torch.ones(
        data.num_edges,
        device=h_index.device,
        dtype=query.dtype,
    )

    if edge_weight_values is None:
        if requires_grad:
            layer_edge_weights = make_independent_edge_weights(
                base_edge_weight, len(model.layers)
            )
        else:
            layer_edge_weights = [base_edge_weight.detach().clone() for _ in model.layers]
    else:
        if len(edge_weight_values) != len(model.layers):
            raise ValueError("edge_weight_values must have one tensor per model layer")
        layer_edge_weights = []
        for value in edge_weight_values:
            if value.shape != base_edge_weight.shape:
                raise ValueError("edge weight override shape mismatch")
            value = value.to(device=h_index.device, dtype=query.dtype)
            if requires_grad:
                value = value.detach().clone().requires_grad_(True)
            else:
                value = value.detach().clone()
            layer_edge_weights.append(value)

    hiddens = []
    layer_input = boundary
    for layer, layer_edge_weight in zip(model.layers, layer_edge_weights):
        hidden = layer(
            layer_input,
            query,
            boundary,
            data.edge_index,
            data.edge_type,
            size,
            layer_edge_weight,
            query_ids=h_index,
        )
        if model.short_cut and hidden.shape == layer_input.shape:
            hidden = hidden + layer_input
        hiddens.append(hidden)
        layer_input = hidden

    node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1)
    if model.concat_hidden:
        output = torch.cat(hiddens + [node_query], dim=-1)
    else:
        output = torch.cat([hiddens[-1], node_query], dim=-1)
    return {"node_feature": output, "edge_weights": layer_edge_weights}


def install_path_attribution_patch(nbfnet_class):
    """Patch only bellmanford(separate_grad=True); ordinary execution is delegated."""
    if not hasattr(nbfnet_class, PATCH_MARKER):
        original = nbfnet_class.bellmanford
        setattr(nbfnet_class, PATCH_MARKER, original)

        def patched_bellmanford(self, data, h_index, r_index, separate_grad=False):
            if not separate_grad:
                return getattr(type(self), PATCH_MARKER)(
                    self, data, h_index, r_index, separate_grad=False
                )
            return _run_independent_bellmanford(
                self, data, h_index, r_index, requires_grad=True
            )

        patched_bellmanford.__name__ = "bellmanford"
        patched_bellmanford.__doc__ = (
            "Compatibility wrapper: original path for separate_grad=False; "
            "independent leaf edge weights for path attribution."
        )
        nbfnet_class.bellmanford = patched_bellmanford
    return nbfnet_class


def _normalize_single_query(batch: torch.Tensor) -> torch.Tensor:
    if batch.shape == (1, 3):
        return batch
    if batch.shape == (1, 1, 3):
        return batch[:, 0, :]
    raise ValueError("single-query batch must have shape [1,3] or [1,1,3]")


def ordinary_query_score(model, data, batch: torch.Tensor) -> torch.Tensor:
    query = _normalize_single_query(batch)
    with torch.inference_mode():
        score = model(data, query.unsqueeze(1)).reshape(-1)
    if score.numel() != 1:
        raise RuntimeError("ordinary single-query inference did not return one score")
    return score[0]


def _score_from_bellmanford_output(model, output, t_index: torch.Tensor):
    feature = output["node_feature"]
    index = t_index.unsqueeze(0).unsqueeze(-1).expand(-1, -1, feature.shape[-1])
    feature = feature.gather(1, index).squeeze(0)
    score = model.mlp(feature).squeeze(-1)
    if score.numel() != 1:
        raise RuntimeError("path-mode single-query inference did not return one score")
    return score.reshape(())


def _gate_parameters(model) -> list[torch.Tensor]:
    values = []
    for layer in model.layers:
        gate = getattr(layer, "dynamic_message_relation_gate", None)
        if gate is None:
            raise RuntimeError("model layer does not expose the frozen RCDGate")
        values.extend(list(gate.parameters()))
    return values


def attribute_query(model, data, batch: torch.Tensor) -> AttributionResult:
    if model.training:
        raise RuntimeError("path attribution requires model.eval()")
    query_batch = _normalize_single_query(batch)
    h_index, t_index, r_index = query_batch.unbind(-1)
    output = model.bellmanford(data, h_index, r_index, separate_grad=True)
    score = _score_from_bellmanford_output(model, output, t_index)
    edge_weights = output["edge_weights"]
    gate_parameters = _gate_parameters(model)
    gradients = torch.autograd.grad(
        score,
        tuple(edge_weights) + tuple(gate_parameters),
        allow_unused=True,
        retain_graph=False,
        create_graph=False,
    )
    edge_gradients = list(gradients[: len(edge_weights)])
    gate_gradients = gradients[len(edge_weights) :]
    if any(value is None for value in edge_gradients):
        raise RuntimeError("score is disconnected from at least one layer edge weight")
    edge_gradients = [value.detach() for value in edge_gradients]
    gate_reachable = bool(gate_gradients) and all(value is not None for value in gate_gradients)
    gate_finite = gate_reachable and all(torch.isfinite(value).all() for value in gate_gradients)
    distances, back_edges = model.beam_search_distance(
        data,
        edge_gradients,
        h_index,
        t_index,
        model.num_beam,
    )
    paths, path_scores = model.topk_average_length(
        distances,
        back_edges,
        t_index,
        model.path_topk,
    )
    return AttributionResult(
        score=score.detach(),
        edge_weights=edge_weights,
        edge_gradients=edge_gradients,
        paths=paths,
        path_scores=path_scores,
        edge_weight_audit=audit_edge_weight_independence(edge_weights),
        gate_gradient_reachable=gate_reachable,
        gate_gradient_finite=gate_finite,
    )


def path_mode_score_with_overrides(
    model,
    data,
    batch: torch.Tensor,
    edge_weight_values: Sequence[torch.Tensor],
) -> torch.Tensor:
    query_batch = _normalize_single_query(batch)
    h_index, t_index, r_index = query_batch.unbind(-1)
    output = _run_independent_bellmanford(
        model,
        data,
        h_index,
        r_index,
        edge_weight_values=edge_weight_values,
        requires_grad=False,
    )
    return _score_from_bellmanford_output(model, output, t_index)


def finite_difference_direction_check(
    model,
    data,
    batch: torch.Tensor,
    attribution: AttributionResult,
    layer_index: int = 0,
    epsilon: float = 1e-3,
) -> dict:
    if not (0 <= layer_index < len(attribution.edge_gradients)):
        raise IndexError("layer_index out of range")
    gradient = attribution.edge_gradients[layer_index]
    edge_id = int(gradient.abs().argmax())
    direction = torch.zeros_like(gradient)
    direction[edge_id] = 1.0
    bases = [torch.ones_like(value) for value in attribution.edge_weights]
    plus = [value.clone() for value in bases]
    minus = [value.clone() for value in bases]
    plus[layer_index] += epsilon * direction
    minus[layer_index] -= epsilon * direction
    with torch.inference_mode():
        score_plus = path_mode_score_with_overrides(model, data, batch, plus)
        score_minus = path_mode_score_with_overrides(model, data, batch, minus)
    finite_difference = float((score_plus - score_minus) / (2 * epsilon))
    autograd_direction = float((gradient * direction).sum())
    absolute_error = abs(finite_difference - autograd_direction)
    tolerance = 5e-3 + 5e-2 * abs(finite_difference)
    return {
        "layer_index": layer_index,
        "edge_index": edge_id,
        "epsilon": epsilon,
        "autograd_directional_derivative": autograd_direction,
        "finite_difference_directional_derivative": finite_difference,
        "absolute_error": absolute_error,
        "tolerance": tolerance,
        "passed": absolute_error <= tolerance,
    }
