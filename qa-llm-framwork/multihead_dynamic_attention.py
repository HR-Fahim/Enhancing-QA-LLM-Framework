import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadDynamicAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=2):  # Reduced heads to 2
        super(MultiHeadDynamicAttention, self).__init__()
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh()
            ) for _ in range(num_heads)
        ])
        self.output_layer = nn.Linear(num_heads * hidden_size, hidden_size)

    def forward(self, x):
        heads_output = [head(x) for head in self.attention_heads]
        combined = torch.cat(heads_output, dim=-1)
        return self.output_layer(combined)

