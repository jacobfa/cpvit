import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaptivePatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_sizes=[4, 8], in_channels=3, embed_dim=64):
        super(AdaptivePatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_sizes = patch_sizes
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.projections = nn.ModuleList()
        # Initialize projections for different patch sizes
        for p in patch_sizes:
            proj = nn.Conv2d(in_channels, embed_dim, kernel_size=p, stride=p)
            self.projections.append(proj)

        # Initialize alpha parameter for weighting embeddings
        self.alpha = nn.Parameter(torch.zeros(len(patch_sizes)))

    def forward(self, x):
        batch_size = x.size(0)
        embeddings = []
        num_patches_list = []

        # Extract embeddings for each patch size
        for proj in self.projections:
            e = proj(x)  # Shape: [B, D, H_p, W_p]
            e = e.flatten(2).transpose(1, 2)  # Shape: [B, N_p, D]
            embeddings.append(e)
            num_patches_list.append(e.size(1))

        max_num_patches = max(num_patches_list)

        # Pad embeddings to have the same sequence length
        padded_embeddings = []
        for e in embeddings:
            pad_length = max_num_patches - e.size(1)
            if pad_length > 0:
                padding = torch.zeros(e.size(0), pad_length, e.size(2), device=e.device, dtype=e.dtype)
                e = torch.cat([e, padding], dim=1)  # Shape: [B, N_max, D]
            padded_embeddings.append(e)

        # Weighted sum of embeddings from different scales
        weights = F.softmax(self.alpha, dim=0)
        combined_embedding = sum(w * e for w, e in zip(weights, padded_embeddings))

        return combined_embedding  # Shape: [B, N_max, D]

class GatedEmbeddingSelection(nn.Module):
    def __init__(self, embed_dim=64):
        super(GatedEmbeddingSelection, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = embed_dim // 2

        self.fc1 = nn.Linear(embed_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        # x: [B, N, D]
        h = F.relu(self.fc1(x))  # [B, N, D']
        g = torch.sigmoid(self.fc2(h))  # [B, N, 1]
        x = x * g  # [B, N, D]
        return x

class MultiHeadAttentionWithAdaptiveScaling(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, gamma=0.5):
        super(MultiHeadAttentionWithAdaptiveScaling, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.gamma = gamma

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x, key_padding_mask=None):
        # x: [B, N, D]
        attn_output, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        scaling_factor = torch.sigmoid(self.beta) * self.gamma
        x = attn_output * scaling_factor
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, max_len=1000, embed_dim=64):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim

        # Create constant 'pe' matrix with values dependent on
        # position and dimension
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                             (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, D]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, N, D]
        x = x + self.pe[:, :x.size(1)]
        return x

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=128, num_classes=10):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class Net(nn.Module):
    # def __init__(self, img_size=32, patch_sizes=[4, 8], in_channels=3, num_classes=10, 
    #              embed_dim=64, num_heads=4, hidden_dim=128):
    def __init__(self, img_size=224, patch_sizes=[8, 16, 32], in_channels=3, num_classes=1000,
                    embed_dim=768, num_heads=12, hidden_dim=3072):
    # def __init__(self, img_size=224, patch_sizes=[8, 16, 32], in_channels=3, num_classes=1000,
    #              embed_dim=1024, num_heads=16, hidden_dim=4096):
    # def __init__(self, img_size=224, patch_sizes=[8, 16, 32], in_channels=3, num_classes=1000,
    #                 embed_dim=1280, num_heads=16, hidden_dim=5120):
        super(Net, self).__init__()
        self.embed_dim = embed_dim

        self.patch_embedding = AdaptivePatchEmbedding(img_size, patch_sizes, 
                                                      in_channels, embed_dim)
        self.gated_selection = GatedEmbeddingSelection(embed_dim)
        self.positional_encoding = PositionalEncoding(
            max_len=(img_size // min(patch_sizes)) ** 2, 
            embed_dim=embed_dim)
        self.attention = MultiHeadAttentionWithAdaptiveScaling(embed_dim, num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.classifier = ClassificationHead(embed_dim, hidden_dim, num_classes)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.patch_embedding(x)  # [B, N_max, D]
        x = self.gated_selection(x)  # [B, N_max, D]
        x = self.positional_encoding(x)  # [B, N_max, D]

        # Optionally create key_padding_mask if you need to mask padded positions
        # key_padding_mask = ...  # [B, N_max], with True at positions that are padded

        x = self.attention(x)  # [B, N_max, D]
        x = x.mean(dim=1)  # Global average pooling over sequence length
        x = self.layer_norm(x)
        x = self.classifier(x)  # [B, num_classes]
        return x
