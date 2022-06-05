import torch
import torch.nn as nn
from tqdm import tqdm

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_size):
        super().__init__()
        assert embed_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.QUERY = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.KEY = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.VALUE = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.linear = nn.Linear(embed_size, embed_size)
    
    def forward(self, query, key, value):
        batch_size, seq_len, _ = query.shape # batchsize, seq_len, embedsize
        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        query = self.QUERY(query)
        key = self.KEY(key)
        value = self.VALUE(value)

        query = query.permute(0, 2, 1, 3) # bz, n_heads, seq_len, h_dim
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        # compute self attention
        t_key = key.transpose(2, 3)
        score = torch.matmul(query, t_key) # bz, n_heads, seq_len, seq_len
        scaled = score / (self.head_dim**0.5)
        softmax = torch.softmax(scaled, dim=-1)
        attention = torch.matmul(softmax, value) # bz, n_heads, seq_len, h_dim
        attention = attention.permute(0, 2, 1, 3) # bz, s_len, n_heads, h_dim
        concat = attention.reshape(attention.size(0), attention.size(1), -1) # bz, seq_len, n_heads * h_dim
        # through linear layer
        out = self.linear(concat)
        return out

# class Embeddings(nn.Module):
#     def __init__(self, embed_size, vocab_size, device):
#         super().__init__()
#         self.input_embedding = nn.Embedding(vocab_size, embed_size)
#         self.embed_size = embed_size
#         self.device = device

#     def forward(self, inputs):
#         batch_size, seq_len = inputs.shape
#         ie = self.input_embedding(inputs)
#         pe = self._get_position_embedding(seq_len, self.embed_size).repeat(batch_size, 1, 1).to(self.device)
#         return ie + pe

#     def _get_position_embedding(self, seq_len, embed_size):
#         pos = torch.arange(seq_len).reshape(1, -1, 1)
#         dim = torch.arange(embed_size).reshape(1, 1, -1)
#         phase = pos / (1e4 ** dim/embed_size)
#         return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


class EncoderLayer(nn.Module):
    def __init__(self, num_heads, embed_size, num_features, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, embed_size)
        self.feed_forward = self._get_feed_forward(embed_size, num_features)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn = self.attention(x, x, x)
        out = x + attn
        out = self.dropout1(self.norm1(out))
        out = out + self.feed_forward(out)
        out = self.dropout2(self.norm2(out))
        return out

    def _get_feed_forward(self, embed_size, num_features):
        return nn.Sequential(
            nn.Linear(embed_size, num_features),
            nn.ReLU(),
            nn.Linear(num_features, embed_size)
        )

# class Encoder(nn.Module):
#     def __init__(
#         self,
#         embed_size=512,
#         vocab_size=1000,
#         device='cpu',
#         num_heads=8,
#         num_features=4*512,
#         num_layers=6,
#         dropout=0.4
#     ) -> None:
#         super().__init__()
#         self.embeddings = Embeddings(embed_size, vocab_size, device)
#         self.encoder_layers = nn.ModuleList([
#             EncoderLayer(num_heads, embed_size, num_features, dropout)
#             for _ in range(num_layers)
#         ])
    
#     def forward(self, input):
#         x = self.embeddings(input)
#         for layer in self.encoder_layers:
#             x = layer(x)
#         return x

class PatchEmbeddings(nn.Module):
    def __init__(self, img_channels, image_size, n_grid, embed_size=768, device='cpu') -> None:
        super().__init__()
        self.patch_size = image_size // n_grid
        self.projection = nn.Conv2d(
            in_channels=img_channels,
            out_channels=embed_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, embed_size))
        self.device = device
        self.embed_size = embed_size

    def forward(self, x):
        batch_size = x.size(0)
        x = self.projection(x)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1) # batch_size, patches, embedsize
        # add cls token at the beginning of each image patch
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        # add positional embedding
        pe = self._get_position_embedding(x.size(1), self.embed_size).repeat(batch_size, 1, 1).to(self.device)
        x += pe
        return x

    def _get_position_embedding(self, seq_len, embed_size):
        pos = torch.arange(seq_len).reshape(1, -1, 1)
        dim = torch.arange(embed_size).reshape(1, 1, -1)
        phase = pos / (1e4 ** dim/embed_size)
        return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

class ViT(nn.Module):
    def __init__(self, img_channels, embed_size, n_grid, img_size, num_classes, num_heads, num_features, num_layers, dropout, device):
        super().__init__()
        self.patch_embedding = PatchEmbeddings(img_channels, img_size, n_grid, embed_size, device)
        self.encoder = nn.ModuleList([
            EncoderLayer(num_heads, embed_size, num_features, dropout)
            for _ in range(num_layers)
        ])
        self.classification_head = nn.Sequential(
            nn.Linear(embed_size, num_classes)
        )
    
    def forward(self, x):
        x = self.patch_embedding(x)
        for layer in self.encoder:
            x = layer(x)
        cls_tokens = x[:, 0]
        out = self.classification_head(cls_tokens)
        return out
