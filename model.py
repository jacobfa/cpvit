import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaptivePatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128, patch_sizes=[4, 8, 16], strides=None):
        super(AdaptivePatchEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.patch_sizes = patch_sizes
        self.n_scales = len(patch_sizes)
        if strides is None:
            # Default strides: half of each patch size
            self.strides = [max(1, pk // 2) for pk in self.patch_sizes]
        else:
            if len(strides) < len(patch_sizes):
                # Extend strides to match the length of patch_sizes
                extra = len(patch_sizes) - len(strides)
                self.strides = strides + [max(1, self.patch_sizes[i + len(strides)] // 2) for i in range(extra)]
            else:
                self.strides = strides

        self.conv_layers = nn.ModuleList()
        for i in range(self.n_scales):
            pk = self.patch_sizes[i]
            sk = self.strides[i]
            conv = nn.Conv2d(in_channels, embed_dim, kernel_size=pk, stride=sk)
            self.conv_layers.append(conv)

    def forward(self, x):
        # x: (B, C, H, W)
        embeddings = []
        positions = []
        B, _, H, W = x.size()
        device = x.device
        for i in range(self.n_scales):
            conv = self.conv_layers[i]
            emb = conv(x)  # (B, D, H', W')
            B, D, Hk, Wk = emb.size()
            emb = emb.view(B, D, -1)  # (B, D, N_k)
            emb = emb.permute(0, 2, 1)  # (B, N_k, D)
            embeddings.append(emb)

            # Compute positions for each patch (optional, can be ignored)
            h_positions = torch.arange(Hk, device=device) * self.strides[i]
            w_positions = torch.arange(Wk, device=device) * self.strides[i]
            grid_y, grid_x = torch.meshgrid(h_positions, w_positions, indexing='ij')
            pos = torch.stack((grid_x, grid_y), dim=-1).view(-1, 2)  # (N_k, 2)
            positions.append(pos)

        return embeddings, positions


class GatedEmbeddingUnit(nn.Module):
    def __init__(self, embed_dim):
        super(GatedEmbeddingUnit, self).__init__()
        D = embed_dim
        D_prime = D // 2
        self.Wg1 = nn.Linear(D, D_prime)
        self.Wg2 = nn.Linear(D_prime, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, e):
        # e: (B, N, D)
        h = self.activation(self.Wg1(e))  # (B, N, D')
        g = self.sigmoid(self.Wg2(h))     # (B, N, 1)
        e_hat = g * e                     # (B, N, D)
        return e_hat


class AggregationProjection(nn.Module):
    def __init__(self, embed_dim):
        super(AggregationProjection, self).__init__()
        D = embed_dim
        self.proj = nn.Linear(D, D)

    def forward(self, e_hat):
        # e_hat: (B, N_total, D)
        # Note: Aggregation per position is approximated by summing over embeddings from all scales
        e_tilde = self.proj(e_hat)    # (B, N_total, D)
        return e_tilde


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.size()
        device = x.device

        # positions shape: [N, 1]
        position = torch.arange(0, N, dtype=torch.float, device=device).unsqueeze(1)

        # div_term shape: [1, D/2]
        div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / D))

        pe = torch.zeros(N, D, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape [1, N, D]
        x = x + pe  # (B, N, D)
        return x


class Net(nn.Module):
    def __init__(self, num_classes=1000, in_channels=3, embed_dim=768,
                 patch_sizes=[4, 8, 16], strides=None, n_heads=12, n_layers=12):
        super(Net, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Adaptive Patch Embedding
        self.patch_embedding = AdaptivePatchEmbedding(
            in_channels, embed_dim, patch_sizes, strides)

        # Gated Embedding Unit
        self.gated_unit = GatedEmbeddingUnit(embed_dim)

        # Aggregation and Projection
        self.aggregation_projection = AggregationProjection(embed_dim)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification Head
        self.layernorm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (B, C, H, W)
        embeddings_list, _ = self.patch_embedding(x)
        # Concatenate embeddings from all scales
        embeddings = torch.cat(embeddings_list, dim=1)  # (B, N_total, D)

        e_hat = self.gated_unit(embeddings)             # (B, N_total, D)
        e_tilde = self.aggregation_projection(e_hat)    # (B, N_total, D)
        e_tilde = self.positional_encoding(e_tilde)     # (B, N_total, D)
        e_tilde = e_tilde.permute(1, 0, 2)              # (N_total, B, D)

        transformer_output = self.transformer_encoder(e_tilde)  # (N_total, B, D)
        attn_output = transformer_output.permute(1, 0, 2)       # (B, N_total, D)

        # Global average pooling over sequence dimension
        z = attn_output.mean(dim=1)                     # (B, D)
        z = self.layernorm(z)
        logits = self.fc(z)                             # (B, num_classes)
        return logits
