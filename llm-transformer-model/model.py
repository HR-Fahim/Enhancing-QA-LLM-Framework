import math
import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        
        # Vextor of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # The sin to be even and the cos to be odd
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # Dimensions: (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpa = nn.Parameter(torch.ones(1)) # Multiplication factor
        self.beta = nn.Parameter(torch.zeros(1)) # Additive factor

    def forwad(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpa * (x - mean) / (std + self.eps) + self.beta


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = self.dropout(torch.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class MultiHeadAttention(nn.Module):
    def ___init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.heads = heads

        assert d_model % heads == 0, "Embedding dimension must be divisible by number of heads"

        self.d_k = d_model // heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch_size, heads, seq_len, d_k)
        attention__scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # (batch_size, heads, seq_len, seq_len)
            attention__scores.masked_fill_(mask == 0, -1e9)

        attention__scores = attention__scores.softmax(dim = -1) # (batch_size, heads, seq_len, seq_len)
        if dropout is not None:
            attention__scores = dropout(attention__scores)

        return (attention__scores @ value), attention__scores
    
    def forward(self, q, k, v, mask):
        # (batch_size, seq_len, d_model): (q, k, v)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch_size, seq_len, heads, d_k)
        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1, 2)

        x, self.attension_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads * self.d_k)

        # (batch_size, seq_len, d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # (batch_size, seq_len, d_model)
        return x + self.dropout(sublayer(self.norm(x))) # Partially modified
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.feed_forward_block = feed_forward_block
        self.self_attention_block = self_attention_block
        self.residual_connection = nn.ModuleList([
            ResidualConnection(dropout) for _ in range(2)
        ])

    def forward(self, x, mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init_(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([
            ResidualConnection(dropout) for _ in range(3)
        ])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch_size, seq_len, vocab_size)
        return self.log_softmax(self.proj(x), dim = -1)
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbeddings, tgt_embedding: InputEmbeddings, src_positional_encoding: PositionalEncoding, tgt_positional_encoding: PositionalEncoding, projection_layer: ProjectionLayer) -> None: 
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_positional_encoding = src_positional_encoding
        self.tgt_positional_encoding = tgt_positional_encoding
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_positional_encoding(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_positional_encoding(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, head: int = 8, d_ff: int = 2, dropout: float = 0.1) -> Transformer:
    src_embedding = InputEmbeddings(src_vocab_size, d_model)
    tgt_embedding = InputEmbeddings(tgt_vocab_size, d_model)
    src_positional_encoding = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_positional_encoding = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    encoder_blocks = []

    for _ in range(N):
        encoder_blocks.append(EncoderBlock(MultiHeadAttention(d_model, head, dropout), FeedForwardBlock(d_model, d_ff, dropout), dropout))

    encoder = Encoder(nn.ModuleList(encoder_blocks))

    decoder_blocks = []

    for _ in range(N):
        decoder_blocks.append(DecoderBlock(MultiHeadAttention(d_model, head, dropout), MultiHeadAttention(d_model, head, dropout), FeedForwardBlock(d_model, d_ff, dropout), dropout))

    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embedding, tgt_embedding, src_positional_encoding, tgt_positional_encoding, projection_layer)

    # Parameters

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer