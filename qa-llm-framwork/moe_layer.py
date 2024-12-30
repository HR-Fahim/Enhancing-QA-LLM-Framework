import torch
import torch.nn as nn

class MoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts=2):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            ) for _ in range(num_experts)
        ])
        self.gating_network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        batch_size, seq_length, hidden_size = x.size()
        x_flat = x.view(-1, hidden_size)
        gate_scores = self.gating_network(x_flat)
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)
        moe_output_flat = torch.sum(gate_scores.unsqueeze(-1) * expert_outputs, dim=1)
        moe_output = moe_output_flat.view(batch_size, seq_length, hidden_size)
        return moe_output
