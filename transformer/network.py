import torch
import torch.nn as nn
import sys
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, n_heads):
        super().__init__()
        assert embed_size % n_heads == 0
        self.head_dim = embed_size // n_heads
        self.embed_size = embed_size
        self.n_heads = n_heads
        self.query_linear = nn.Linear(self.head_dim, self.head_dim)
        self.key_linear = nn.Linear(self.head_dim, self.head_dim)
        self.value_linear = nn.Linear(self.head_dim, self.head_dim)
        self.linear_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, query, key, value, mask=None):
        batch_size =  query.shape[0]
        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]
        query = query.reshape(batch_size, query_len, self.n_heads, self.head_dim)
        key = key.reshape(batch_size, key_len, self.n_heads, self.head_dim)
        value = value.reshape(batch_size, value_len, self.n_heads, self.head_dim)
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)
        query = query.permute(0, 2, 1, 3) # bs, heads, qlen, dim
        key = key.permute(0, 2, 1, 3) # bs, heads, klen, dim
        value = value.permute(0, 2, 1, 3) # bs, heads, vlen, dim
        # self attention
        score = torch.matmul(query, key.transpose(-1, -2)) # bs, heads, qlen, klen
        scaled = score / (self.head_dim ** 0.5)
        if mask is not None:
            scaled.masked_fill_(mask==0, -1e5)
        softmax = torch.softmax(scaled, dim=-1)
        attention = torch.matmul(softmax, value) # bs, heads, qlen, dim
        attention = attention.permute(0, 2, 1, 3) # bs, qlen, heads, dim
        attention = attention.reshape(batch_size, query_len, -1)
        attention = self.linear_out(attention)
        return attention
    
class Embeddings(nn.Module):
    def __init__(self, embed_size, max_length, vocab_size, device):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.device = device
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        # seq_len = x.shape[1]
        # pos = torch.arange(seq_len, dtype=torch.long, device=self.device)
        # pos = pos.unsqueeze(0) # 1, slen
        # pos = pos.expand_as(x) # bs, slen
        # ie = self.ie(x) # bs, slen, embed_size
        # pe = self.pe(pos) # bs, slen, embed_size
        # embedding = ie + pe        N, seq_length = x.shape
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
        return x

class EncoderLayer(nn.Module):
    def __init__(self, embed_size, n_heads, dropout, expansion):
        super().__init__()
        self.attention_layer = MultiHeadAttention(embed_size, n_heads)
        self.feed_forward = self._feed_forward(embed_size, expansion)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
    
    def forward(self, x, mask):
        x = x + self.attention_layer(x, x, x, mask)
        x = self.dropout1(self.norm1(x))
        x = x + self.feed_forward(x)
        x = self.dropout2(self.norm2(x))
        return x
        
    def _feed_forward(self, in_features, expansion):
        return nn.Sequential(
            nn.Linear(in_features, expansion*in_features),
            nn.ReLU(),
            nn.Linear(expansion*in_features, in_features)
        )
    
class Encoder(nn.Module):
    def __init__(
        self,
        embed_size,
        n_heads,
        dropout,
        vocab_size,
        expansion,
        n_layers,
        max_length,
        device
    ):
        super().__init__()
        self.embeddings = Embeddings(embed_size, max_length, vocab_size, device)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embed_size, n_heads, dropout, expansion)
            for _ in range(n_layers)
        ])
    
    def forward(self, src, src_mask):
        x = self.embeddings(src)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, embed_size, n_heads, dropout, expansion) -> None:
        super().__init__()
        self.attention_layer1 = MultiHeadAttention(embed_size, n_heads)
        self.attention_layer2 = MultiHeadAttention(embed_size, n_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.feed_forward = self._feed_forward(embed_size, expansion)

    def forward(self, x, context, src_mask, trg_mask):
        """
        context is the output from encoder
        """

        x = x + self.attention_layer1(x, x, x, trg_mask)
        x = self.dropout1(self.norm1(x))
        x = x + self.attention_layer2(x, context, context, src_mask)
        x = self.dropout2(self.norm2(x))
        x = x + self.feed_forward(x)
        x = self.dropout3(self.norm3(x))
        return x

    def _feed_forward(self, in_features, expansion):
        return nn.Sequential(
            nn.Linear(in_features, expansion*in_features),
            nn.ReLU(),
            nn.Linear(expansion*in_features, in_features)
        )

class Decoder(nn.Module):
    def __init__(
        self,
        embed_size,
        n_heads,
        dropout,
        vocab_size,
        expansion,
        n_layers,
        max_length,
        device
    ) -> None:
        super().__init__()
        self.embedding = Embeddings(embed_size, max_length, vocab_size, device)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embed_size, n_heads, dropout, expansion) \
                for _ in range(n_layers)
        ])
        self.classifer = nn.Linear(embed_size, vocab_size, bias=False)
    
    def forward(self, x, context, src_mask, trg_mask):
        x = self.embedding(x)
        for layer in self.decoder_layers:
            x = layer(x, context, src_mask, trg_mask)
        x = self.classifer(x)
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size=100,
        trg_vocab_size=100,
        src_max_length=25,
        trg_max_length=25,
        embed_size=512,
        n_heads=8,
        encode_layers=6,
        decode_layers=6,
        expansion=4,
        dropout=0,
        device='cpu'
    ) -> None:
        super().__init__()
        self.encoder = Encoder(embed_size, n_heads, dropout, src_vocab_size, expansion, encode_layers, src_max_length, device)
        self.decoder = Decoder(embed_size, n_heads, dropout, trg_vocab_size, expansion, decode_layers, trg_max_length, device)
        self.device = device
    
    def forward(self, src, trg):
        src_mask = self._make_src_mask(src)
        trg_mask = self._make_trg_mask(trg)
        context = self.encoder(src, src_mask)
        out = self.decoder(trg, context, src_mask, trg_mask)
        return out

    def _make_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def _make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)
