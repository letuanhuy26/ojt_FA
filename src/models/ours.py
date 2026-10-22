from __future__ import annotations

import torch
from torch import nn


class FeatureGateLSTM(nn.Module):
    """LSTM controls for testing incremental value of regime-conditioned gates."""

    MODES = {
        "ungated_lstm",
        "static_gate_lstm",
        "dynamic_gate_lstm",
        "regime_as_feature_lstm",
        "regime_dynamic_gate_lstm",
    }

    def __init__(self, n_features: int, n_regimes: int, hidden_size: int = 32,
                 gate_hidden_size: int = 32, dropout: float = 0.0,
                 mode: str = "regime_dynamic_gate_lstm"):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown model mode {mode}; allowed={sorted(self.MODES)}")
        self.mode = mode
        self.n_features = n_features
        self.n_regimes = n_regimes
        if mode in {"dynamic_gate_lstm", "regime_dynamic_gate_lstm"}:
            cond_size = n_features + (n_regimes if mode == "regime_dynamic_gate_lstm" else 0)
            self.gate_net = nn.Sequential(
                nn.Linear(cond_size, gate_hidden_size),
                nn.ReLU(),
                nn.Linear(gate_hidden_size, n_features),
                nn.Sigmoid(),
            )
        elif mode == "static_gate_lstm":
            self.static_gate_logits = nn.Parameter(torch.zeros(n_features))
        lstm_input = n_features + (n_regimes if mode == "regime_as_feature_lstm" else 0)
        self.lstm = nn.LSTM(lstm_input, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "regime_dynamic_gate_lstm":
            gates = self.gate_net(torch.cat([x, p], dim=-1))
            model_input = x * gates
        elif self.mode == "dynamic_gate_lstm":
            gates = self.gate_net(x)
            model_input = x * gates
        elif self.mode == "static_gate_lstm":
            gates = torch.sigmoid(self.static_gate_logits).view(1, 1, -1).expand_as(x)
            model_input = x * gates
        else:
            gates = torch.ones_like(x)
            model_input = torch.cat([x, p], dim=-1) if self.mode == "regime_as_feature_lstm" else x
        sequence, _ = self.lstm(model_input)
        logits = self.output(self.dropout(sequence[:, -1])).squeeze(-1)
        return logits, gates


def loss_with_gate_penalties(logits: torch.Tensor, y: torch.Tensor, gates: torch.Tensor,
                             lambda_sparsity: float, lambda_stability: float) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
    if lambda_sparsity:
        loss = loss + lambda_sparsity * gates.mean()
    if lambda_stability and gates.shape[1] > 1:
        loss = loss + lambda_stability * (gates[:, 1:] - gates[:, :-1]).abs().mean()
    return loss
