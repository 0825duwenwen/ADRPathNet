from __future__ import annotations

import torch
from torch import nn


class ZeroLinear(nn.Module):
    """Linear map whose construction consumes no random numbers."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(value, self.weight, self.bias)


class ReceiverCompatibilityDynamicGate(nn.Module):
    """Scalar DGate augmented only by message/receiver compatibility.

    z = w_m^T m + w_r^T r + w_c^T (m * h_receiver) + b
    g = 1 + 0.5 tanh(z)
    """

    def __init__(self, hidden_dim: int, gate_scale: float = 0.5):
        super().__init__()
        if float(gate_scale) <= 0:
            raise ValueError("gate_scale must be positive")
        self.hidden_dim = int(hidden_dim)
        self.gate_scale = float(gate_scale)
        # These names preserve the matched DGate checkpoint/state contract.
        self.message_linear = ZeroLinear(self.hidden_dim, 1, bias=True)
        self.relation_linear = ZeroLinear(self.hidden_dim, 1, bias=False)
        # Explicit zeros: +hidden_dim parameters and no global RNG consumption.
        self.compatibility_weight = nn.Parameter(torch.zeros(self.hidden_dim))
        self.statistics_observer = None

    def compute_gate(self, message, relation, receiver):
        expected = message.shape
        if message.ndim != 3 or relation.shape != expected or receiver.shape != expected:
            raise RuntimeError("RCDGate expects matching [batch, edge, hidden] tensors")
        if message.shape[-1] != self.hidden_dim:
            raise RuntimeError("RCDGate hidden dimension mismatch")
        compatibility = message * receiver
        compatibility_logit = torch.sum(
            compatibility * self.compatibility_weight, dim=-1, keepdim=True
        )
        logit = (
            self.message_linear(message)
            + self.relation_linear(relation)
            + compatibility_logit
        )
        gate = 1.0 + self.gate_scale * torch.tanh(logit)
        if self.statistics_observer is not None:
            self.statistics_observer(gate.detach(), compatibility_logit.detach())
        return logit, gate

    def forward(self, message, relation, receiver):
        _, gate = self.compute_gate(message, relation, receiver)
        return message * gate

    def forward_with_details(self, message, relation, receiver):
        logit, gate = self.compute_gate(message, relation, receiver)
        return message * gate, logit, gate
