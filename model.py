import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

# Constants for ImageNet (Hyperparameters maintained)
NUM_CLASSES = 1000
IMAGE_SIZE = 224
PATCH_SIZES = [16, 32]  # Multi-scale patches
EMBED_DIM = 768
NUM_HEADS = 12
NUM_LAYERS = 12
HIDDEN_DIM = 3072
DROPOUT = 0.1
GRAPH_LAYER_DIM = 768
GRAPH_HEADS = 8
GRAPH_LAYERS = 3

class MultiScalePatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_sizes, in_chans, embed_dim):
        super(MultiScalePatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_sizes = patch_sizes
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.projections = nn.ModuleList()
        for P in patch_sizes:
            proj = nn.Conv2d(in_chans, embed_dim, kernel_size=P, stride=P)
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
            self.projections.append(proj)

        # Layer Normalization after patch embeddings
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.size(0)
        patch_embeddings = []
        positions = []
        for proj, P in zip(self.projections, self.patch_sizes):
            x_patches = proj(x)  # B x C x H_p x W_p
            B, C, H_p, W_p = x_patches.shape
            x_patches = x_patches.flatten(2).transpose(1, 2)  # B x N x C

            # Layer Norm
            x_patches = self.norm(x_patches)

            # Generate positional coordinates as FloatTensor
            x_coords = torch.arange(W_p, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(H_p, 1)  # H_p x W_p
            y_coords = torch.arange(H_p, device=x.device, dtype=x.dtype).unsqueeze(1).repeat(1, W_p)  # H_p x W_p
            pos = torch.stack([x_coords, y_coords], dim=-1).view(-1, 2)  # (N_p, 2)
            pos = pos.unsqueeze(0).repeat(B, 1, 1)  # B x N_p x 2

            patch_embeddings.append(x_patches)  # B x N_p x C
            positions.append(pos)  # B x N_p x 2
        return patch_embeddings, positions

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=10000):
        super(SinusoidalPositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # (max_len, dim)

    def forward(self, positions):
        # positions: (N, 2)
        N = positions.size(0)
        positions_flat = positions[:, 0] * 1000 + positions[:, 1]  # Simple flattening
        positions_flat = positions_flat.long()
        pe = self.pe[positions_flat]  # (N, dim)
        return pe

class HybridGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, beta=0.1, dropout=DROPOUT):
        super(HybridGATConv, self).__init__(aggr='add')  # 'add' aggregation
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dropout = nn.Dropout(dropout)

        self.W = nn.Linear(in_channels, out_channels, bias=False)
        nn.init.xavier_uniform_(self.W.weight)

        self.a = nn.Parameter(torch.empty(2 * out_channels, 1))
        nn.init.xavier_uniform_(self.a)

        self.beta = nn.Parameter(torch.tensor(beta))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        # Layer Normalization after GNN
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        x = self.W(x)  # [N, out_channels]

        # Prepare index pointers for source and target nodes
        row, col = edge_index  # edge_index has shape [2, E]

        x_i = x[row]  # [E, out_channels]
        x_j = x[col]  # [E, out_channels]

        # Concatenate x_i and x_j
        x_cat = torch.cat([x_i, x_j], dim=-1)  # [E, 2 * out_channels]

        # Compute dot product
        dot_product = (x_i * x_j).sum(dim=-1, keepdim=True)  # [E, 1]

        # Compute e_{ij}
        e_cat = torch.matmul(x_cat, self.a)  # [E, 1]
        e_ij = self.leaky_relu(e_cat + self.beta * dot_product)  # [E, 1]

        # Compute attention coefficients
        alpha = softmax(e_ij, index=row)  # [E, 1]
        alpha = self.dropout(alpha)

        # Apply attention coefficients to node features in message passing
        return self.propagate(edge_index, x=x, alpha=alpha)

    def message(self, x_j, alpha):
        return alpha * x_j  # Both have shape [E, out_channels]

    def update(self, aggr_out):
        # Apply layer normalization
        return self.norm(aggr_out)

class GraphAttentionNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, heads):
        super(GraphAttentionNetwork, self).__init__()
        self.gat_layers = nn.ModuleList()
        self.num_layers = num_layers

        # First layer
        self.gat_layers.append(
            HybridGATConv(in_dim, hidden_dim)
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                HybridGATConv(hidden_dim, hidden_dim)
            )

        # Output layer
        self.gat_layers.append(
            HybridGATConv(hidden_dim, out_dim)
        )

        self.out_dim = out_dim

        # Layer Norm after GNN
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index):
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)  # Use ReLU activation
        x = self.norm(x)  # Apply Layer Norm after GNN
        return x

class GraphConditionedTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, graph_dim):
        super(GraphConditionedTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=DROPOUT)
        self.graph_proj = nn.Linear(graph_dim, embed_dim)
        nn.init.xavier_uniform_(self.graph_proj.weight)
        nn.init.zeros_(self.graph_proj.bias)
        self.linear1 = nn.Linear(embed_dim, HIDDEN_DIM)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        self.dropout = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(HIDDEN_DIM, embed_dim)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.activation = F.gelu

    def forward(self, src, graph_features, src_mask=None, src_key_padding_mask=None):
        # Graph-conditioned attention
        graph_emb = self.graph_proj(graph_features)
        q = k = src + graph_emb
        src2, _ = self.self_attn(q, k, src, attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        # Feed-forward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, graph_dim):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            GraphConditionedTransformerEncoderLayer(embed_dim, num_heads, graph_dim)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, src, graph_features, src_mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, graph_features, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return output

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, num_classes)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x

class Net(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, image_size=IMAGE_SIZE,
                 patch_sizes=PATCH_SIZES, in_chans=3, embed_dim=EMBED_DIM,
                 num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
                 graph_layer_dim=GRAPH_LAYER_DIM, graph_heads=GRAPH_HEADS, graph_layers=GRAPH_LAYERS):
        super(Net, self).__init__()
        self.embed_dim = embed_dim
        self.patch_embedding = MultiScalePatchEmbedding(image_size, patch_sizes, in_chans, embed_dim)

        # Sinusoidal Positional Embedding
        self.pos_emb = SinusoidalPositionalEmbedding(embed_dim)

        # [CLS] token for Transformer
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=.02)

        # [CLS] token for GNN features
        self.cls_token_gnn = nn.Parameter(torch.zeros(1, graph_layer_dim))
        nn.init.trunc_normal_(self.cls_token_gnn, std=.02)

        # Edge attention parameters for dynamic graph construction
        self.edge_attention_a = nn.Parameter(torch.empty(2 * embed_dim, 1))
        nn.init.xavier_uniform_(self.edge_attention_a)
        self.edge_attention_beta = nn.Parameter(torch.tensor(0.1))  # initialize beta to 0.1

        # Graph Attention Network
        self.gnn = GraphAttentionNetwork(in_dim=embed_dim, hidden_dim=graph_layer_dim,
                                         out_dim=graph_layer_dim, num_layers=graph_layers,
                                         heads=graph_heads)

        # Transformer Encoder
        self.transformer_encoder = TransformerEncoder(embed_dim=embed_dim, num_layers=num_layers,
                                                      num_heads=num_heads, graph_dim=graph_layer_dim)

        # Classification Head
        self.classifier = ClassificationHead(embed_dim, num_classes)

    def forward(self, x):
        # x: (B, C, H, W)
        B = x.size(0)

        # Multi-scale patch embedding
        patch_embeddings_list, positions_list = self.patch_embedding(x)

        # Initialize lists to store per-sample data
        data_list = []

        for i in range(B):  # For each sample in batch
            x_patches_i = []
            positions_i = []
            for emb, pos in zip(patch_embeddings_list, positions_list):
                x_patches_i.append(emb[i])  # N_p x D
                positions_i.append(pos[i])   # N_p x 2

            x_patches_i = torch.cat(x_patches_i, dim=0)  # N_i x D
            positions_i = torch.cat(positions_i, dim=0)  # N_i x 2

            N_i = x_patches_i.size(0)
            positions_i = positions_i.float()

            # Normalize positions to [0,1]
            positions_i = positions_i / (positions_i.max() + 1e-6)

            # Compute positional embeddings and add to patch embeddings
            pos_embeds_i = self.pos_emb(positions_i)  # (N_i, D)
            x_patches_i = x_patches_i + pos_embeds_i  # (N_i, D)

            x_i = x_patches_i  # (N_i, D)

            # Compute dot product similarities
            dot_product = torch.matmul(x_i, x_i.T)  # (N_i, N_i)

            # Expand x_i and x_j to (N_i * N_i, 2D)
            x_i_expanded = x_i.unsqueeze(1).expand(-1, N_i, -1)  # (N_i, N_i, D)
            x_j_expanded = x_i.unsqueeze(0).expand(N_i, -1, -1)  # (N_i, N_i, D)
            x_cat = torch.cat([x_i_expanded, x_j_expanded], dim=-1)  # (N_i, N_i, 2D)
            x_cat = x_cat.view(N_i * N_i, -1)  # (N_i * N_i, 2D)

            # Compute e_{ij}
            a = self.edge_attention_a  # (2D, 1)
            e_cat = torch.matmul(x_cat, a).view(N_i, N_i)  # (N_i * N_i, 1) -> (N_i, N_i)
            beta = self.edge_attention_beta  # scalar
            e_ij = F.leaky_relu(e_cat + beta * dot_product)  # (N_i, N_i)

            # Set diagonal elements (self-loops) to -inf to avoid selecting them
            e_ij = e_ij.clone()
            e_ij.fill_diagonal_(-float('inf'))

            K = min(8, N_i - 1)  # Adjust K, ensure it's less than N_i
            topk_values, topk_indices = torch.topk(e_ij, K, dim=1)  # (N_i, K)

            # Build edge_index
            row_indices = torch.arange(N_i, device=x.device).unsqueeze(1).expand(N_i, K).flatten()  # (N_i * K,)
            col_indices = topk_indices.flatten()  # (N_i * K,)

            edge_index = torch.stack([row_indices, col_indices], dim=0)  # (2, N_i * K)

            # Create Data object
            data = Data(x=x_patches_i, pos=positions_i, edge_index=edge_index)
            data_list.append(data)

        # Batch the data
        batch = Batch.from_data_list(data_list)

        # GNN forward
        gnn_features = self.gnn(batch.x, batch.edge_index)  # (total_N_nodes, D_g)

        # Extract the batch information
        batch_indices = batch.batch  # (total_N_nodes,)

        # Prepare per-sample sequences for the transformer
        src_list = []
        graph_features_list = []
        seq_lengths = []
        for i in range(B):
            mask = (batch_indices == i)
            src_i = batch.x[mask]  # N_i x D
            gnn_features_i = gnn_features[mask]  # N_i x D_g

            # Add [CLS] token to src_i and gnn_features_i
            cls_token_expanded = self.cls_token  # (1, D)
            src_i = torch.cat([cls_token_expanded, src_i], dim=0)  # (N_i + 1, D)

            cls_token_gnn_expanded = self.cls_token_gnn  # (1, D_g)
            gnn_features_i = torch.cat([cls_token_gnn_expanded, gnn_features_i], dim=0)  # (N_i + 1, D_g)

            seq_len = src_i.size(0)
            seq_lengths.append(seq_len)

            src_list.append(src_i)
            graph_features_list.append(gnn_features_i)

        max_seq_len = max(seq_lengths)

        # Pad sequences to max length
        src_padded = torch.zeros(B, max_seq_len, self.embed_dim, device=x.device)
        gnn_padded = torch.zeros(B, max_seq_len, self.gnn.out_dim, device=x.device)
        src_key_padding_mask = torch.ones(B, max_seq_len, dtype=torch.bool, device=x.device)

        for i in range(B):
            seq_len = src_list[i].size(0)
            src_padded[i, :seq_len, :] = src_list[i]
            gnn_padded[i, :seq_len, :] = graph_features_list[i]
            src_key_padding_mask[i, :seq_len] = False  # False where we have actual data

        # Transpose to match the expected input shape N x B x D
        src = src_padded.transpose(0, 1)  # N x B x D
        graph_features = gnn_padded.transpose(0, 1)  # N x B x D_g

        # Transformer Encoder
        output = self.transformer_encoder(src, graph_features, src_key_padding_mask=src_key_padding_mask)

        # Classification Head using [CLS] token
        output = output.transpose(0, 1)  # B x N x D
        cls_output = output[:, 0, :]  # B x D

        logits = self.classifier(cls_output)

        return logits
