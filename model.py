import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=64):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        return x  # [B, embed_dim, grid_size, grid_size]

class MultiScalePatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_sizes, in_channels=3, embed_dim=64):
        super(MultiScalePatchEmbedding, self).__init__()
        self.patch_sizes = patch_sizes
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.grid_sizes = [img_size // p for p in patch_sizes]
        self.H_max = max(self.grid_sizes)
        self.W_max = max(self.grid_sizes)
        self.N_max = self.H_max * self.W_max

        self.embeddings = nn.ModuleList([
            PatchEmbedding(img_size, p, in_channels, embed_dim)
            for p in patch_sizes
        ])

    def forward(self, x):
        # x: [B, C, H, W]
        B = x.size(0)
        embeddings_aligned = []
        for i, embedding_layer in enumerate(self.embeddings):
            embeddings = embedding_layer(x)  # [B, D, H_k, W_k]
            H_k, W_k = embeddings.size(2), embeddings.size(3)
            if H_k == self.H_max and W_k == self.W_max:
                embeddings_aligned.append(embeddings)  # [B, D, H_max, W_max]
            else:
                # Expand embeddings to [B, D, H_max, W_max]
                factor_H = self.H_max // H_k
                factor_W = self.W_max // W_k

                embeddings_expanded = embeddings.repeat_interleave(factor_H, dim=2)
                embeddings_expanded = embeddings_expanded.repeat_interleave(factor_W, dim=3)
                embeddings_aligned.append(embeddings_expanded)  # [B, D, H_max, W_max]

        # Concatenate along the channel dimension
        embeddings_concat = torch.cat(embeddings_aligned, dim=1)  # [B, K*D, H_max, W_max]
        # Flatten and transpose to [B, N_max, K*D]
        embeddings_concat = embeddings_concat.flatten(2).transpose(1, 2)  # [B, N_max, K*D]
        return embeddings_concat  # [B, N_max, K*D]

class MultiScaleEmbeddingProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MultiScaleEmbeddingProjection, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # x: [B, N_max, K*D]
        x = self.proj(x)  # [B, N_max, D]
        return x

class GatedEmbeddingSelection(nn.Module):
    def __init__(self, embed_dim):
        super(GatedEmbeddingSelection, self).__init__()
        D = embed_dim
        D_prime = D // 2
        self.fc1 = nn.Linear(D, D_prime)
        self.fc2 = nn.Linear(D_prime, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, N_max, D]
        h = self.activation(self.fc1(x))  # [B, N_max, D']
        g = self.sigmoid(self.fc2(h))  # [B, N_max, 1]
        x = g * x  # [B, N_max, D]
        return x

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        # x: [B, N_max, D]
        x = x.transpose(0, 1)  # [N_max, B, D]
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.transpose(0, 1)  # [B, N_max, D]
        return x

class PositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim, height, width):
        super(PositionalEncoding2D, self).__init__()
        self.embed_dim = embed_dim
        self.height = height
        self.width = width

        pe = self.create_positional_encoding(embed_dim, height, width)
        self.register_buffer('pe', pe)

    def create_positional_encoding(self, embed_dim, height, width):
        pe = torch.zeros(height * width, embed_dim)
        y_pos, x_pos = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing='ij'
        )
        y_pos = y_pos.flatten().unsqueeze(1)  # [H*W, 1]
        x_pos = x_pos.flatten().unsqueeze(1)  # [H*W, 1]

        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin((x_pos + y_pos) * div_term)
        pe[:, 1::2] = torch.cos((x_pos + y_pos) * div_term)
        return pe  # [H*W, D]

    def forward(self, x):
        # x: [B, N_max, D]
        x = x + self.pe.unsqueeze(0).to(x.device)  # [B, N_max, D]
        return x

class Net(nn.Module):
    # def __init__(
    #    self,
    #     img_size=32,
    #     patch_sizes=[4, 8],
    #     in_channels=3,
    #     num_classes=10,
    #     embed_dim=64,
    #     num_heads=4 
    # ):
    
    def __init__(self, img_size=224, patch_sizes=[8, 16, 32], in_channels=3, num_classes=1000, embed_dim=768, num_heads=12):
    # def __init__(self, img_size=224, patch_sizes=[8, 16, 32], in_channels=3, num_classes=1000, embed_dim=1024, num_heads=16):
    # def __init__(self, img_size=224, patch_sizes=[8, 16, 32], in_channels=3, num_classes=1000, embed_dim=1280, num_heads=16):
        super(Net, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.hidden_dim = embed_dim * 2

        self.patch_embedding = MultiScalePatchEmbedding(
            img_size=img_size,
            patch_sizes=patch_sizes,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        self.projection = MultiScaleEmbeddingProjection(
            in_dim=len(patch_sizes) * embed_dim,
            out_dim=embed_dim
        )
        self.gated_embedding = GatedEmbeddingSelection(embed_dim)
        self.positional_encoding = PositionalEncoding2D(
            embed_dim=embed_dim,
            height=img_size // min(patch_sizes),
            width=img_size // min(patch_sizes)
        )
        self.self_attention = SelfAttention(embed_dim, num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.patch_embedding(x)            # [B, N_max, K*D]
        x = self.projection(x)                # [B, N_max, D]
        x = self.gated_embedding(x)           # [B, N_max, D]
        x = self.positional_encoding(x)       # [B, N_max, D]
        x = self.self_attention(x)            # [B, N_max, D]
        x = self.layer_norm(x)                # [B, N_max, D]
        x = x.mean(dim=1)                     # Global average pooling [B, D]
        x = self.classifier(x)                # [B, num_classes]
        return x
