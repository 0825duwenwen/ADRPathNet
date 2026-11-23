"""Gate modules that only depend on PyTorch.

Keeping DGate here allows its math, gradients and capture policy to be tested
without importing PyG or torch_scatter.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn


class DynamicMessageRelationGate(nn.Module):
    """Message-state-conditioned scalar gate with an identity initialization."""

    def __init__(self, hidden_dim, gate_scale=0.5):
        super().__init__()
        if gate_scale <= 0:
            raise ValueError("DGate gate_scale must be greater than 0")
        if gate_scale != 0.5:
            raise ValueError("NBFNet-DGate uses the fixed gate_scale=0.5")
        self.hidden_dim = int(hidden_dim)
        self.gate_scale = float(gate_scale)
        self.message_linear = nn.Linear(self.hidden_dim, 1, bias=True)
        self.relation_linear = nn.Linear(self.hidden_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.message_linear.weight)
        nn.init.zeros_(self.message_linear.bias)
        nn.init.zeros_(self.relation_linear.weight)

    def compute_gate(self, message, relation):
        self._validate_inputs(message, relation)
        logit = self.message_linear(message) + self.relation_linear(relation)
        gate = 1.0 + self.gate_scale * torch.tanh(logit)
        return logit, gate

    def forward(self, message, relation):
        _, gate = self.compute_gate(message, relation)
        return message * gate

    def forward_with_details(self, message, relation):
        """Explicit analysis-only API; normal forward never retains gate tensors."""
        logit, gate = self.compute_gate(message, relation)
        return message * gate, logit, gate

    def _validate_inputs(self, message, relation):
        if message.ndim != 3 or relation.ndim != 3:
            raise RuntimeError(
                "DGate expects [batch, edge, hidden] tensors, got message=%s relation=%s"
                % (tuple(message.shape), tuple(relation.shape))
            )
        if message.shape != relation.shape:
            raise RuntimeError(
                "DGate message/relation shapes must match, got %s and %s"
                % (tuple(message.shape), tuple(relation.shape))
            )
        if message.shape[-1] != self.hidden_dim:
            raise RuntimeError(
                "DGate hidden dimension mismatch: expected %d, got %d"
                % (self.hidden_dim, message.shape[-1])
            )


class DGateCapture:
    """Bounded, opt-in CPU capture for selected queries and top-k edges."""

    def __init__(self, query_drug_ids, topk=100, layer_ids=None, histogram_bins=20):
        query_drug_ids = {int(query_id) for query_id in query_drug_ids}
        if not query_drug_ids:
            raise ValueError("DGate capture requires at least one query drug ID")
        if topk <= 0:
            raise ValueError("DGate capture topk must be greater than 0")
        if histogram_bins <= 0:
            raise ValueError("DGate histogram_bins must be greater than 0")
        self.query_drug_ids = query_drug_ids
        self.topk = int(topk)
        self.layer_ids = None if layer_ids is None else {int(i) for i in layer_ids}
        self.histogram_bins = int(histogram_bins)
        self.records = []
        self._stats = defaultdict(
            lambda: {
                "count": 0,
                "sum": 0.0,
                "sum_sq": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
                "histogram": torch.zeros(self.histogram_bins, dtype=torch.long),
            }
        )

    @torch.no_grad()
    def capture(
        self,
        layer_id,
        query_ids,
        edge_index,
        edge_type,
        raw_message,
        gate_logit,
        gate_value,
    ):
        if self.layer_ids is not None and int(layer_id) not in self.layer_ids:
            return
        selected_rows = [
            row
            for row, query_id in enumerate(query_ids.detach().cpu().tolist())
            if int(query_id) in self.query_drug_ids
        ]
        for row in selected_rows:
            row_gate = gate_value[row, :, 0].detach()
            row_logit = gate_logit[row, :, 0].detach()
            row_norm = raw_message[row].detach().norm(dim=-1)
            num_selected = min(self.topk, row_gate.numel())
            edge_ids = (row_gate - 1.0).abs().topk(num_selected).indices
            query_id = int(query_ids[row].detach().cpu())
            for edge_id in edge_ids.detach().cpu().tolist():
                self.records.append(
                    {
                        "query_drug": query_id,
                        "layer": int(layer_id),
                        "edge_id": int(edge_id),
                        "source_entity": int(edge_index[0, edge_id].detach().cpu()),
                        "target_entity": int(edge_index[1, edge_id].detach().cpu()),
                        "relation": int(edge_type[edge_id].detach().cpu()),
                        "raw_message_norm": float(row_norm[edge_id].cpu()),
                        "gate_logit": float(row_logit[edge_id].cpu()),
                        "gate_value": float(row_gate[edge_id].cpu()),
                    }
                )
            for relation_id in edge_type.unique().detach().cpu().tolist():
                relation_mask = edge_type == int(relation_id)
                values = row_gate[relation_mask]
                if values.numel():
                    key = (int(layer_id), int(relation_id))
                    stats = self._stats[key]
                    cpu_values = values.float().cpu()
                    stats["count"] += cpu_values.numel()
                    stats["sum"] += float(cpu_values.sum())
                    stats["sum_sq"] += float((cpu_values ** 2).sum())
                    stats["min"] = min(stats["min"], float(cpu_values.min()))
                    stats["max"] = max(stats["max"], float(cpu_values.max()))
                    stats["histogram"] += torch.histc(
                        cpu_values,
                        bins=self.histogram_bins,
                        min=0.5,
                        max=1.5,
                    ).long()

    def summaries(self):
        result = []
        for (layer_id, relation_id), stats in sorted(self._stats.items()):
            count = stats["count"]
            mean = stats["sum"] / count
            variance = max(stats["sum_sq"] / count - mean * mean, 0.0)
            result.append(
                {
                    "layer": layer_id,
                    "relation": relation_id,
                    "count": count,
                    "mean": mean,
                    "standard_deviation": variance ** 0.5,
                    "min": stats["min"],
                    "max": stats["max"],
                    "histogram_range": [0.5, 1.5],
                    "histogram": stats["histogram"].tolist(),
                }
            )
        return result
