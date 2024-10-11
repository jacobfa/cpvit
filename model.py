import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptivePatchEmbedding(nn.Module):
    """
    Learnable dynamic combination of patch embeddings from different patch sizes
    using linear projections to align embeddings to a consistent sequence length.
    """
    def __init__(self, img_size=32, patch_sizes=(2, 4, 8), embed_dim=256):
        super(AdaptivePatchEmbedding, self).__init__()
        self.patch_sizes = patch_sizes
        self.embed_dim = embed_dim
        self.img_size = img_size

        # Calculate number of patches for each patch size
        self.num_patches_dict = {}
        for patch_size in patch_sizes:
            assert img_size % patch_size == 0, f"Image size {img_size} must be divisible by patch size {patch_size}"
            num_patches = (img_size // patch_size) ** 2
            self.num_patches_dict[str(patch_size)] = num_patches

        # Determine the maximum number of patches
        self.N_max = max(self.num_patches_dict.values())

        # Create a convolution layer for each possible patch size
        self.embeddings = nn.ModuleDict({
            str(patch_size): nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
            for patch_size in patch_sizes
        })

        # Learnable weights for combining embeddings from different patch sizes
        self.alpha = nn.Parameter(torch.zeros(len(patch_sizes)))  # For softmax weights

        # Define linear projections for patch sizes with N_k < N_max
        self.projections = nn.ModuleDict()
        for patch_size in patch_sizes:
            N_k = self.num_patches_dict[str(patch_size)]
            if N_k < self.N_max:
                # Define a linear layer to map N_k to N_max for each D dimension
                # Including bias for better learning
                self.projections[str(patch_size)] = nn.Linear(N_k, self.N_max, bias=True)
            else:
                # No projection needed
                self.projections[str(patch_size)] = nn.Identity()

        # Initialize projection layers
        for proj in self.projections.values():
            if isinstance(proj, nn.Linear):
                nn.init.xavier_uniform_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)

    def forward(self, x):
        batch_size = x.size(0)
        embeddings = []

        # Compute embeddings for all patch sizes
        for patch_size in self.patch_sizes:
            proj_key = str(patch_size)
            embed_layer = self.embeddings[proj_key].to(x.device)
            emb = embed_layer(x)  # Shape: [B, D, H_p, W_p]
            H_p, W_p = emb.shape[2], emb.shape[3]
            N_k = H_p * W_p
            assert N_k == self.num_patches_dict[proj_key], f"Expected {self.num_patches_dict[proj_key]} patches, but got {N_k}"
            emb = emb.flatten(2).transpose(1, 2)  # Shape: [B, N_k, D]

            # Apply linear projection if necessary
            proj_layer = self.projections[proj_key]
            if isinstance(proj_layer, nn.Linear):
                # Reshape to [B * D, N_k]
                emb_reshaped = emb.permute(0, 2, 1).contiguous()  # [B, D, N_k]
                emb_reshaped = emb_reshaped.view(batch_size * self.embed_dim, N_k)  # [B*D, N_k]
                emb_proj = proj_layer(emb_reshaped)  # [B*D, N_max]
                emb_proj = emb_proj.view(batch_size, self.embed_dim, self.N_max)  # [B, D, N_max]
                emb_proj = emb_proj.transpose(1, 2).contiguous()  # [B, N_max, D]
                embeddings.append(emb_proj)
            else:
                # Identity projection
                embeddings.append(emb)  # [B, N_max, D]

        # Compute weights via softmax
        weights = F.softmax(self.alpha, dim=0)  # Shape: [K]

        # Stack embeddings: [K, B, N_max, D]
        stacked_embeddings = torch.stack(embeddings, dim=0)  # [K, B, N_max, D]

        # Apply weights: [B, N_max, D]
        weights = weights.view(-1, 1, 1, 1)  # [K, 1, 1, 1]
        combined_embedding = (weights * stacked_embeddings).sum(dim=0)  # [B, N_max, D]

        # Apply LayerNorm for better training stability
        combined_embedding = F.layer_norm(combined_embedding, combined_embedding.shape[-1:])

        return combined_embedding  # Shape: [B, N_max, embed_dim]

class GatedPatchSelection(nn.Module):
    """
    A gating mechanism that learns to select important patches.
    """
    def __init__(self, embed_dim):
        super(GatedPatchSelection, self).__init__()
        D_prime = embed_dim // 2
        self.gating_network = nn.Sequential(
            nn.Linear(embed_dim, D_prime),
            nn.ReLU(),
            nn.Linear(D_prime, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Compute gate values for each patch
        gate_scores = self.gating_network(x).squeeze(-1)  # Shape: [B, N_max]
        gated_output = x * gate_scores.unsqueeze(-1)  # Shape: [B, N_max, embed_dim]
        return gated_output

class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention heads to capture details at different resolutions,
    with adaptive scaling of attention scores.
    """
    def __init__(self, embed_dim, num_heads, num_scales=3, scaling_factor=0.5):
        super(MultiScaleAttention, self).__init__()
        self.num_scales = num_scales
        self.scaling_factor = scaling_factor

        self.attention_heads = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in range(num_scales)
        ])

        # Learnable scaling factors for each scale
        self.beta = nn.Parameter(torch.ones(num_scales))

        self.gamma = scaling_factor  # Predefined constant scaling factor

        # Projection after concatenation
        self.projection = nn.Linear(embed_dim * num_scales, embed_dim)

        # Initialize projection layer
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x):
        multi_scale_outputs = []
        for s in range(self.num_scales):
            attn_head = self.attention_heads[s]
            attn_output, attn_weights = attn_head(x, x, x)  # attn_output: [B, N_max, D]
            # Apply adaptive scaling
            adaptive_scale = torch.sigmoid(self.beta[s]) * self.gamma  # Scalar
            attn_output_scaled = attn_output * adaptive_scale  # [B, N_max, D]
            multi_scale_outputs.append(attn_output_scaled)

        # Concatenate outputs from all scales
        concatenated = torch.cat(multi_scale_outputs, dim=-1)  # [B, N_max, D * num_scales]

        # Project back to original embed_dim
        projected_output = self.projection(concatenated)  # [B, N_max, D]

        # Apply LayerNorm for better training stability
        projected_output = F.layer_norm(projected_output, projected_output.shape[-1:])

        return projected_output

class TransformerBlock(nn.Module):
    """
    Transformer block with adaptive components.
    """
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super(TransformerBlock, self).__init__()
        self.attn = MultiScaleAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # Initialize MLP layers
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Net(nn.Module):
    """
    The main model with dynamic positional encoding generation, optimized for classification.
    """
    # def __init__(self, img_size=32, patch_sizes=(2, 4, 8), embed_dim=256, num_heads=8, mlp_dim=512, num_layers=6, num_classes=100):
    def __init__(self, img_size=224, patch_sizes=(4, 8, 16), embed_dim=768, num_heads=12, mlp_dim=3072, num_layers=12, num_classes=1000):
    # def __init__(self, img_size=224, patch_sizes=(8, 16, 32), embed_dim=1024, num_heads=16, mlp_dim=4096, num_layers=12, num_classes=1000):
    # def __init__(self, img_size=224, patch_sizes=(8, 16, 32), embed_dim=1280, num_heads=16, mlp_dim=5120, num_layers=12, num_classes=1000):
        super(Net, self).__init__()
        self.patch_embedding = AdaptivePatchEmbedding(img_size, patch_sizes, embed_dim)
        self.embed_dim = embed_dim
        self.N_max = self.patch_embedding.N_max  # Ensure consistency

        self.gated_selection = GatedPatchSelection(embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        # Initialize classification head
        for m in self.mlp_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Adaptive Patch Embedding with Linear Projection
        E = self.patch_embedding(x)  # Shape: [B, N_max, D]
        batch_size, N_max, D = E.size()

        # Dynamic Positional Encoding
        pos_encoding = self.generate_positional_encoding(N_max, D, device=x.device)  # [1, N_max, D]
        X = E + pos_encoding  # [B, N_max, D]

        # Gated Patch Selection
        X = self.gated_selection(X)  # [B, N_max, D]

        # Transformer Blocks
        for block in self.transformer_blocks:
            X = block(X)  # [B, N_max, D]

        # Global Pooling and Classification Head
        z = X.mean(dim=1)  # [B, D]
        logits = self.mlp_head(z)  # [B, C]

        return logits

    def generate_positional_encoding(self, num_patches, embed_dim, device):
        """
        Generate a sinusoidal positional encoding dynamically based on the number of patches.
        """
        position = torch.arange(0, num_patches, dtype=torch.float, device=device).unsqueeze(1)  # [N_max, 1]
        div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        
        pos_enc = torch.zeros((num_patches, embed_dim), device=device)
        pos_enc[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pos_enc[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        return pos_enc.unsqueeze(0)  # [1, N_max, D]
