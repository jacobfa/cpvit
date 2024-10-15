import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=64, patch_size=16, stride=8, in_channels=3, embed_dim=256):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim

        self.conv = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=0
        )

    def forward(self, x):
        x = self.conv(x)  # [B, D, H', W']
        x = x.flatten(2)  # [B, D, N]
        x = x.transpose(1, 2)  # [B, N, D]
        return x  # [B, N, D]

class GatedEmbeddingUnit(nn.Module):
    def __init__(self, embed_dim):
        super(GatedEmbeddingUnit, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim // 2)
        self.fc2 = nn.Linear(embed_dim // 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.ReLU()

    def forward(self, x):
        h = self.activation(self.fc1(x))  # [B, N, D']
        g = self.sigmoid(self.fc2(h))  # [B, N, 1]
        x = x * g  # [B, N, D]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, num_positions, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))

    def forward(self, x):
        x = x + self.pos_embed
        return x

class MultiScaleEmbedding(nn.Module):
    def __init__(self, img_size=64, patch_sizes=[4, 8, 16], strides=[2, 4, 8], in_channels=3, embed_dim=256):
        super(MultiScaleEmbedding, self).__init__()
        self.embed_layers = nn.ModuleList([
            PatchEmbedding(img_size, p, s, in_channels, embed_dim) for p, s in zip(patch_sizes, strides)
        ])
        self.gate_unit = GatedEmbeddingUnit(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        embeddings = []
        positions = []
        batch_size = x.size(0)
        device = x.device

        max_num_patches = 0
        embeddings_list = []
        positions_list = []

        for embed_layer in self.embed_layers:
            emb = embed_layer(x)  # [B, N_k, D]
            N_k = emb.size(1)
            embeddings_list.append(emb)
            positions_list.append(N_k)
            if N_k > max_num_patches:
                max_num_patches = N_k

        aligned_embeddings = torch.zeros(batch_size, max_num_patches, len(self.embed_layers), self.embed_dim, device=device)
        for idx, emb in enumerate(embeddings_list):
            N_k = emb.size(1)
            aligned_embeddings[:, :N_k, idx, :] = emb

        # Flatten M_i dimension
        B, N, M_i, D = aligned_embeddings.size()
        aligned_embeddings = aligned_embeddings.view(B * N, M_i, D)

        # Apply gating mechanism
        aligned_embeddings = self.gate_unit(aligned_embeddings)  # [B*N, M_i, D]
        # Aggregate embeddings
        aggregated_embeddings = aligned_embeddings.sum(dim=1)  # [B*N, D]
        aggregated_embeddings = self.proj(aggregated_embeddings)  # [B*N, D]

        # Reshape back to [B, N, D]
        aggregated_embeddings = aggregated_embeddings.view(batch_size, max_num_patches, self.embed_dim)

        return aggregated_embeddings  # [B, N, D]

class Net(nn.Module):
    def __init__(
        self,
        img_size=64,
        patch_sizes=[4, 8, 16],
        strides=[2, 4, 8],
        in_channels=3,
        num_classes=200,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1
    ):
        super(Net, self).__init__()
        self.embed_dim = embed_dim

        self.embedding = MultiScaleEmbedding(
            img_size=img_size,
            patch_sizes=patch_sizes,
            strides=strides,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        # Positional Encoding
        num_positions = max(
            [
                ((img_size - p) // s + 1)
                * ((img_size - p) // s + 1)
                for p, s in zip(patch_sizes, strides)
            ]
        )
        self.pos_encoder = PositionalEncoding(num_positions, embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.embedding(x)  # [B, N, D]
        x = self.pos_encoder(x)  # [B, N, D]

        # Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, D]

        # Transformer expects input shape [N+1, B, D]
        x = x.transpose(0, 1)  # [N+1, B, D]

        x = self.transformer_encoder(x)  # [N+1, B, D]

        # Take the class token output
        cls_output = x[0]  # [B, D]

        cls_output = self.layer_norm(cls_output)  # [B, D]
        logits = self.fc(cls_output)  # [B, num_classes]

        return logits
