import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=64):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N_k, D]
        return x

class GatedEmbeddingSelection(nn.Module):
    def __init__(self, embed_dim):
        super(GatedEmbeddingSelection, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim // 2)
        self.fc2 = nn.Linear(embed_dim // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, N, D]
        h = F.relu(self.fc1(x))  # [B, N, D//2]
        g = self.sigmoid(self.fc2(h))  # [B, N, 1]
        x = x * g  # Element-wise multiplication
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, D]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, N, D]
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

class Net(nn.Module):
    # def __init__(
    #     self,
    #     img_size=32,
    #     in_channels=3,
    #     num_classes=10,
    #     embed_dim=64,
    #     patch_sizes=[4, 8],
    #     num_heads=4,
    #     num_layers=2
    # ):
    
    def __init__(
        self,
        img_size=224,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        patch_sizes=[8, 16, 32],
        num_heads=12,
        num_layers=12
    ):
        
        super(Net, self).__init__()
        self.embed_dim = embed_dim

        # Multi-scale patch embeddings
        self.patch_embeddings = nn.ModuleList([
            PatchEmbedding(
                img_size=img_size,
                patch_size=p,
                in_channels=in_channels,
                embed_dim=embed_dim
            )
            for p in patch_sizes
        ])

        # Gated embedding selection
        self.gated_selection = GatedEmbeddingSelection(embed_dim=embed_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim=embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        embeddings = []
        for embed in self.patch_embeddings:
            e = embed(x)  # [B, N_k, D]
            embeddings.append(e)

        # Concatenate embeddings from all scales
        x = torch.cat(embeddings, dim=1)  # [B, N_total, D]

        # Gated embedding selection
        x = self.gated_selection(x)  # [B, N_total, D]

        # Positional encoding
        x = self.pos_encoder(x)  # [B, N_total, D]

        # Transformer encoder expects input shape [N_total, B, D]
        x = x.permute(1, 0, 2)  # [N_total, B, D]

        # Self-attention
        x = self.transformer_encoder(x)  # [N_total, B, D]

        # Convert back to [B, N_total, D]
        x = x.permute(1, 0, 2)

        # Global average pooling
        x = x.mean(dim=1)  # [B, D]

        # Classification head
        x = self.classifier(x)  # [B, num_classes]
        return x
