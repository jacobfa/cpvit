import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph

# Constants for ImageNet
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


class GraphAttentionNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, heads):
        super(GraphAttentionNetwork, self).__init__()
        self.gat_layers = nn.ModuleList()
        self.num_layers = num_layers

        # Ensure consistent dimensions for residual connections
        assert in_dim == hidden_dim == out_dim, "For residual connections, in_dim, hidden_dim, and out_dim must be equal"

        # GAT layers with residual connections
        for _ in range(num_layers):
            self.gat_layers.append(
                GATConv(in_dim, in_dim, heads=heads, concat=False, dropout=DROPOUT)
            )

        self.out_dim = out_dim

        # Layer Norm after GNN
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index):
        for gat_layer in self.gat_layers:
            residual = x
            x = gat_layer(x, edge_index)
            x = x + residual  # Residual connection
            x = F.relu(x)
        x = self.norm(x)  # Apply Layer Norm after GNN
        return x


class GraphConditionedTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, graph_dim, drop_path=0.0):
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

        # LayerScale parameters
        self.layer_scale_1 = nn.Parameter(torch.ones(embed_dim) * 0.1)
        self.layer_scale_2 = nn.Parameter(torch.ones(embed_dim) * 0.1)

        # Stochastic Depth (DropPath)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Removed relative positional encoding for now

    def forward(self, src, graph_features, src_mask=None, src_key_padding_mask=None):
        # Pre-Layer Norm
        src2 = self.norm1(src)
        graph_emb = self.graph_proj(graph_features)
        q = k = src2 + graph_emb
        v = src2

        # Multi-head Attention
        attn_output, _ = self.self_attn(q, k, v, attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask)
        # Apply LayerScale and DropPath
        src = src + self.drop_path(self.layer_scale_1 * attn_output)

        # Feed-forward network
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))

        # Apply LayerScale and DropPath
        src = src + self.drop_path(self.layer_scale_2 * src2)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, graph_dim):
        super(TransformerEncoder, self).__init__()
        dpr = [x.item() for x in torch.linspace(0, 0.1, num_layers)]  # Stochastic depth decay rule
        self.layers = nn.ModuleList([
            GraphConditionedTransformerEncoderLayer(embed_dim, num_heads, graph_dim, drop_path=dpr[i])
            for i in range(num_layers)
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


class DropPath(nn.Module):
    """
    DropPath class implements Stochastic Depth as per the original paper.

    Reference: https://arxiv.org/abs/1603.09382
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)

    @staticmethod
    def drop_path(x, drop_prob=0.0, training=False):
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # Work with tensors of any dimensionality
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class Net(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, image_size=IMAGE_SIZE,
                 patch_sizes=PATCH_SIZES, in_chans=3, embed_dim=EMBED_DIM,
                 num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
                 graph_layer_dim=GRAPH_LAYER_DIM, graph_heads=GRAPH_HEADS, graph_layers=GRAPH_LAYERS):
        super(Net, self).__init__()
        self.embed_dim = embed_dim
        self.patch_embedding = MultiScalePatchEmbedding(image_size, patch_sizes, in_chans, embed_dim)

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional embedding layer (absolute positional embeddings)
        self.pos_embedding = nn.Linear(2, embed_dim)
        nn.init.xavier_uniform_(self.pos_embedding.weight)
        nn.init.zeros_(self.pos_embedding.bias)

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

            # Ensure positions_i is a FloatTensor
            positions_i = positions_i.float()

            # Build edge_index for sample i
            # Build a k-NN graph based on positions_i
            k = 8  # You can adjust k to control the sparsity
            edge_index = knn_graph(positions_i, k=k, batch=None, loop=False)

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
        max_seq_len = 0
        for i in range(B):
            mask = (batch_indices == i)
            src_i = batch.x[mask]  # N_i x D
            gnn_features_i = gnn_features[mask]  # N_i x D_g
            positions_i = batch.pos[mask]  # N_i x 2

            # Map positions to positional embeddings
            pos_emb_i = self.pos_embedding(positions_i)  # N_i x D

            # Add positional embeddings to src_i
            src_i = src_i + pos_emb_i  # N_i x D

            # Prepend cls_token
            cls_token_expanded = self.cls_token.expand(1, -1, -1).squeeze(0)  # 1 x D
            src_i = torch.cat([cls_token_expanded, src_i], dim=0)  # (N_i + 1) x D

            # For graph_features, we initialize a zero cls token
            graph_cls_token = torch.zeros(1, self.gnn.out_dim, device=x.device)  # 1 x D_g
            gnn_features_i = torch.cat([graph_cls_token, gnn_features_i], dim=0)  # (N_i + 1) x D_g

            seq_len = src_i.size(0)
            max_seq_len = max(max_seq_len, seq_len)

            src_list.append(src_i)
            graph_features_list.append(gnn_features_i)

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

        # Classification Head
        # Use the output corresponding to the cls_token
        output = output.transpose(0, 1)  # B x N x D
        cls_output = output[:, 0, :]  # B x D

        logits = self.classifier(cls_output)

        return logits
