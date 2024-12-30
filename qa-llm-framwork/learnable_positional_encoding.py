import torch
import torch.nn as nn

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, hidden_size):
        super(LearnablePositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, hidden_size))
        nn.init.xavier_uniform_(self.positional_encoding)

    def forward(self, x):
        pos_encoding = self.positional_encoding[:, :x.size(1), :]
        return x + pos_encoding
