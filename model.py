import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_sizes=[2, 4, 8], strides=[1, 2, 4], in_channels=3, embed_dim=512):
        super(PatchEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.convs = nn.ModuleList()
        for p, s in zip(patch_sizes, strides):
            self.convs.append(
                nn.Conv2d(in_channels, embed_dim, kernel_size=p, stride=s, padding=p // 2)
            )
        self.num_patches_list = []
        for p, s in zip(patch_sizes, strides):
            num_patches = ((img_size + s - 1) // s) ** 2
            self.num_patches_list.append(num_patches)
        self.N_max = max(self.num_patches_list)

    def forward(self, x):
        embeddings_list = []
        for conv in self.convs:
            emb = conv(x)  # [B, D, H_p, W_p]
            B, D, H_p, W_p = emb.shape
            emb = emb.flatten(2).transpose(1, 2)  # [B, N_k, D]
            embeddings_list.append(emb)
        return embeddings_list  # List of embeddings at different scales


class GatedEmbeddingSelection(nn.Module):
    def __init__(self, embed_dim):
        super(GatedEmbeddingSelection, self).__init__()
        D = embed_dim
        D_prime = D * 2
        self.Wg1 = nn.Linear(D, D_prime)
        self.Wg2 = nn.Linear(D_prime, D_prime)
        self.Wg3 = nn.Linear(D_prime, 1)
        self.proj = nn.Linear(D, D)
        self.activation = nn.GELU()

    def forward(self, embeddings_list):
        N_max = max([emb.size(1) for emb in embeddings_list])
        embeddings_padded = []
        for emb in embeddings_list:
            B, N_k, D = emb.size()
            if N_k < N_max:
                pad_size = N_max - N_k
                padding = torch.zeros(B, pad_size, D, device=emb.device, dtype=emb.dtype)
                emb_padded = torch.cat([emb, padding], dim=1)  # [B, N_max, D]
            else:
                emb_padded = emb
            embeddings_padded.append(emb_padded)
        embeddings_stack = torch.stack(embeddings_padded, dim=2)  # [B, N_max, num_scales, D]
        B, N_max, num_scales, D = embeddings_stack.size()
        embeddings = embeddings_stack.view(B * N_max * num_scales, D)
        h = self.activation(self.Wg1(embeddings))  # [B * N_max * num_scales, D']
        h = self.activation(self.Wg2(h))  # Additional layer
        g = torch.sigmoid(self.Wg3(h))  # [B * N_max * num_scales, 1]
        embeddings_gated = embeddings * g  # [B * N_max * num_scales, D]
        embeddings_gated = embeddings_gated.view(B, N_max, num_scales, D)
        aggregated_embeddings = embeddings_gated.sum(dim=2)  # [B, N_max, D]
        aggregated_embeddings = self.proj(aggregated_embeddings)  # [B, N_max, D]
        return aggregated_embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        seq_len = x.size(1)
        device = x.device
        position = torch.arange(0, seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, device=device).float() * (-math.log(10000.0) / self.embed_dim))
        pe = torch.zeros(seq_len, self.embed_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # [1, seq_len, embed_dim]
        x = x + pe
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, src):
        src2 = self.attention(src, src, src)[0]
        src = src + src2
        src = self.layernorm1(src)
        src2 = self.mlp(src)
        src = src + src2
        src = self.layernorm2(src)
        return src


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.fc(x)
        return x


class Net(nn.Module):
    # def __init__(self, img_size=32, patch_sizes=[2, 4, 8], strides=[1, 2, 4], in_channels=3,
    #              embed_dim=128, num_heads=8, num_layers=12, mlp_dim=128*4, num_classes=10, dropout=0.1):
    def __init__(self, img_size=32, patch_sizes=[4, 8 , 16], strides=[2, 4, 8], in_channels=3,
                 embed_dim=768, num_heads=12, num_layers=12, mlp_dim=768*4, num_classes=1000, dropout=0.1):
    
        super(Net, self).__init__()
        self.embed_dim = embed_dim
        self.patch_embedding = PatchEmbedding(img_size, patch_sizes, strides, in_channels, embed_dim)
        self.gated_embedding = GatedEmbeddingSelection(embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.classification_head = ClassificationHead(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        embeddings_list = self.patch_embedding(x)
        embeddings = self.gated_embedding(embeddings_list)  # [B, N_max, D]
        embeddings = self.positional_encoding(embeddings)
        embeddings = embeddings.transpose(0, 1)  # [N_max, B, D]
        for layer in self.transformer_layers:
            embeddings = layer(embeddings)
        embeddings = embeddings.transpose(0, 1)  # [B, N_max, D]
        z = embeddings.mean(dim=1)  # [B, D]
        logits = self.classification_head(z)
        return logits
