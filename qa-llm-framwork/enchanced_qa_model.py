import torch
import torch.nn as nn

from learnable_positional_encoding import LearnablePositionalEncoding
from multihead_dynamic_attention import MultiHeadDynamicAttention
from moe_layer import MoELayer

class EnhancedQAModel(nn.Module):
    def __init__(self, model_class, hidden_size):
        super(EnhancedQAModel, self).__init__()
        self.model = model_class()
        self.hidden_size = hidden_size
        self.positional_encoding = LearnablePositionalEncoding(max_len=512, hidden_size=self.hidden_size)
        self.multi_head_attention = MultiHeadDynamicAttention(self.hidden_size, num_heads=2)
        self.moe_layer = MoELayer(self.hidden_size, num_experts=2)
        self.qa_outputs = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 2)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        sequence_output = self.positional_encoding(sequence_output)
        sequence_output = self.multi_head_attention(sequence_output)
        sequence_output = self.moe_layer(sequence_output)
        logits = self.qa_outputs(sequence_output)
        return logits
