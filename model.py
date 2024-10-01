import torch
import torch.nn as nn
import torch.nn.functional as F

# Base Module
class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

# Patch Embedding Module
class PatchEmbedding(BaseModule):
    def __init__(self, in_channels, embed_dim, patch_size, stride):
        super(PatchEmbedding, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )

    def forward(self, x):
        x = self.conv(x)  # [B, embed_dim, H_p, W_p]
        H_p, W_p = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, N_k, embed_dim]
        return x, H_p, W_p

# Gated Embedding Selection Module
class GatedEmbeddingSelection(BaseModule):
    def __init__(self, embed_dim):
        super(GatedEmbeddingSelection, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim // 2)
        self.fc2 = nn.Linear(embed_dim // 2, 1)

    def forward(self, embeddings):
        h = F.relu(self.fc1(embeddings))
        g = torch.sigmoid(self.fc2(h))
        embeddings_gated = embeddings * g  # Element-wise multiplication
        return embeddings_gated

# Positional Encoding Module
class PositionalEncoding(BaseModule):
    def __init__(self, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.position_embedding = None

    def forward(self, positions):
        if self.position_embedding is None:
            pe = self._get_positional_encoding(positions)
            self.position_embedding = pe
        else:
            pe = self.position_embedding
        return pe

    def _get_positional_encoding(self, positions):
        device = positions.device
        N_max = positions.size(0)
        pe = torch.zeros(N_max, self.embed_dim, device=device)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, device=device) *
            (-torch.log(torch.tensor(10000.0, device=device)) / self.embed_dim)
        )
        pos_x = positions[:, 0].unsqueeze(1)
        pos_y = positions[:, 1].unsqueeze(1)
        pe[:, 0::2] = torch.sin(pos_x * div_term)
        pe[:, 1::2] = torch.cos(pos_y * div_term)
        return pe

# Multi-Scale Transformer Module
class MultiScaleTransformer(BaseModule):
    def __init__(self, embed_dim, num_heads, dropout):
        super(MultiScaleTransformer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, embeddings):
        embeddings_transposed = embeddings.transpose(0, 1)
        attn_output, _ = self.attention(embeddings_transposed, embeddings_transposed, embeddings_transposed)
        attn_output = attn_output.transpose(0, 1)
        attn_output = self.layer_norm(attn_output)
        return attn_output

# Final Classification Module
class ClassificationHead(BaseModule):
    def __init__(self, embed_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        logits = self.classifier(x)
        return logits

# Main Network
class Net(BaseModule):
    def __init__(
        self,
        num_classes=1000,
        image_size=224,
        patch_sizes=[4, 8, 16],
        strides=[4, 8, 16],
        embed_dim=768,
        num_heads=12,
        dropout=0.1,
    ):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.patch_sizes = patch_sizes
        self.strides = strides

        # Modules
        self.patch_embeddings = nn.ModuleList([
            PatchEmbedding(3, embed_dim, p, s) for p, s in zip(patch_sizes, strides)
        ])
        self.gate = GatedEmbeddingSelection(embed_dim)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim)
        self.transformer = MultiScaleTransformer(embed_dim, num_heads, dropout)
        self.classification_head = ClassificationHead(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        device = x.device

        embeddings_list = []
        positions_list = []
        for pe in self.patch_embeddings:
            e, H_p, W_p = pe(x)
            embeddings_list.append(e)

            pos_y = torch.arange(H_p, device=device).float() * (self.image_size / H_p)
            pos_x = torch.arange(W_p, device=device).float() * (self.image_size / W_p)
            grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing='ij')
            positions = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
            positions_list.append(positions)

        N_max = max(e.size(1) for e in embeddings_list)
        embeddings_aligned = []
        positions_aligned = []

        for e, positions in zip(embeddings_list, positions_list):
            N_k = e.size(1)
            if N_k < N_max:
                repeat_factor = N_max // N_k + 1
                e = e.repeat(1, repeat_factor, 1)[:, :N_max, :]
                positions = positions.repeat(repeat_factor, 1)[:N_max, :]
            embeddings_aligned.append(e)
            positions_aligned.append(positions)

        embeddings_stack = torch.stack(embeddings_aligned, dim=2)
        positions_stack = torch.stack(positions_aligned, dim=0)

        embeddings_flat = embeddings_stack.reshape(-1, self.embed_dim)

        embeddings_gated = self.gate(embeddings_flat)

        embeddings_gated = embeddings_gated.view(B, N_max, len(self.patch_sizes), self.embed_dim)

        embeddings_aggregated = embeddings_gated.sum(dim=2)

        embeddings_projected = self.projection(embeddings_aggregated)

        positions_mean = torch.mean(positions_stack, dim=0)
        pe = self.position_encoding(positions_mean)

        embeddings_positioned = embeddings_projected + pe.unsqueeze(0)

        attn_output = self.transformer(embeddings_positioned)

        z = attn_output.mean(dim=1)

        logits = self.classification_head(z)

        return logits
