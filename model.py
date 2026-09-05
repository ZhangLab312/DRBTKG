"""模型定义 / Model definitions."""
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv, Linear

"""任务定义 / Task definition"""
node_type1 = "drug"
node_type2 = "disease"
rel = "indication"


def configure_model_context(**context):
    globals().update(context)


class FixedSemanticEmbeddingManager:
    """语义嵌入管理器 / Semantic embedding manager."""
    def __init__(self, cfg):
        self.config = cfg
        self.semantic_embeddings = None
        self.semantic_mappings = None
        self.entity_to_embedding = {}

        self.load_semantic_data()

    def load_semantic_data(self):
        try:
            self.semantic_embeddings = torch.load(
                self.config["semantic_embedding_path"],
                map_location=device,
            )

            print(
                "Semantic embedding shape: "
                f"{self.semantic_embeddings.shape}"
            )

        except Exception as e:
            print(f"Failed to load semantic embeddings: {e}")
            return

        try:
            with open(self.config["semantic_mapping_path"], "rb") as file:
                self.semantic_mappings = pickle.load(file)

            if "metadata" not in self.semantic_mappings:
                return

            for i, meta in enumerate(self.semantic_mappings["metadata"]):
                if i >= len(self.semantic_embeddings):
                    break

                for key in (
                    meta.get("drugbank_id", ""),
                    meta.get("drug_name", ""),
                ):
                    key = str(key).strip().lower()
                    if key:
                        self.entity_to_embedding[key] = self.semantic_embeddings[i]

            print(
                f"Semantic mappings: {len(self.entity_to_embedding)} drug keys"
            )

        except Exception as e:
            print(f"Failed to load semantic mappings: {e}")

    def get_semantic_embedding(self, entity_name):
        if not entity_name:
            return None

        clean_name = str(entity_name).strip().lower()

        if clean_name in self.entity_to_embedding:
            return self.entity_to_embedding[clean_name]

        return None


class SemanticProjectionNetwork(nn.Module):
    """语义投影网络 / Semantic projection network."""
    def __init__(
        self,
        semantic_embedding_dim,
        projected_embedding_dim,
        dropout=0.2,
    ):
        super().__init__()

        # 嵌入投影层 / Embedding projection layers.
        self.projection = nn.Sequential(
            Linear(
                semantic_embedding_dim,
                projected_embedding_dim * 2,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(
                projected_embedding_dim * 2,
                projected_embedding_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, semantic_embeddings):
        return self.projection(semantic_embeddings)


class ProjectionEnhancedHGAT(nn.Module):
    """异构图注意力网络 / Heterogeneous graph attention network."""
    def __init__(
        self,
        node_feature_dims,
        hidden_channels,
        out_channels,
        num_heads,
        num_layers,
        dropout,
        message_passing_graph,
        projected_embedding_dim=64,
        use_semantic_projection=True,
    ):
        super().__init__()

        self.use_semantic_projection = use_semantic_projection
        self.projected_embedding_dim = projected_embedding_dim
        self.feature_fusion = nn.ModuleDict()

        for node_type, feature_dim in node_feature_dims.items():
            fusion_input_dim = feature_dim

            self.feature_fusion[node_type] = nn.Sequential(
                Linear(fusion_input_dim, hidden_channels[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.convs = nn.ModuleList()

        # 各关系的图注意力层 / Relation-specific GAT layers.
        for layer_id in range(num_layers):
            conv_dict = {}

            for edge_type in message_passing_graph.edge_types:
                conv_dict[edge_type] = GATConv(
                    in_channels=hidden_channels[layer_id],
                    out_channels=(
                        hidden_channels[layer_id + 1]
                        // num_heads[layer_id]
                    ),
                    heads=num_heads[layer_id],
                    dropout=dropout,
                    concat=True,
                    add_self_loops=False,
                )

            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        self.self_loops = nn.ModuleDict()
        for node_type in message_passing_graph.node_types:
            self.self_loops[node_type] = nn.Linear(
                hidden_channels[-1],
                hidden_channels[-1],
            )

        self.lin = Linear(sum(hidden_channels[1:]), out_channels)
        self.dropout = nn.Dropout(dropout)

    def get_projected_embeddings_for_nodes(
        self,
        x_dict,
        batch,
        projection_net,
    ):
        projected_dict = {}

        if not self.use_semantic_projection:
            return projected_dict

        for node_type in {"drug", "disease"}:
            if node_type not in x_dict:
                continue

            x = x_dict[node_type]
            projected_embeddings = torch.zeros(
                x.size(0), self.projected_embedding_dim, device=x.device
            )

            if (
                node_type == "drug"
                and projection_net is not None
                and hasattr(batch["drug"], "text_embedding")
            ):
                projected_embeddings = projection_net(
                    batch["drug"].text_embedding.to(x.device)
                )

            projected_dict[node_type] = projected_embeddings

        return projected_dict

    def forward(self, x_dict, edge_index_dict, batch, projection_net):
        projected_dict = self.get_projected_embeddings_for_nodes(
            x_dict,
            batch,
            projection_net,
        )
        fused_x_dict = {}

        for node_type, x in x_dict.items():
            if node_type in projected_dict:
                x = torch.cat([x, projected_dict[node_type]], dim=1)
            fused_x_dict[node_type] = self.feature_fusion[node_type](x)

        x_dict = fused_x_dict
        out = {}

        for conv in self.convs:
            x_dict_conv = conv(x_dict, edge_index_dict)
            x_dict = {}
            for node_type, node_x in x_dict_conv.items():
                self_loop = self.self_loops[node_type](node_x)
                x_dict[node_type] = F.relu(node_x + self_loop)

            # 拼接各层输出 / Concatenate layer outputs.
            if not out:
                out = copy.copy(x_dict)
            else:
                out = {
                    node_type: torch.cat(
                        [out[node_type], x_dict[node_type]],
                        dim=1,
                    )
                    for node_type in x_dict
                }

        return (
            F.relu(self.lin(out[node_type1])),
            F.relu(self.lin(out[node_type2])),
        )


class EnhancedMLPPredictor(nn.Module):
    """链接预测器 / Link predictor."""
    def __init__(self, channel_num, dropout):
        super().__init__()
        self.L1 = nn.Linear(channel_num * 2, channel_num * 2)
        self.L2 = nn.Linear(channel_num * 2, channel_num)
        self.L3 = nn.Linear(channel_num, 1)
        self.bn1 = nn.BatchNorm1d(channel_num * 2)
        self.bn2 = nn.BatchNorm1d(channel_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug_embeddings, disease_embeddings):
        x = torch.cat([drug_embeddings, disease_embeddings], dim=1)
        x = self.dropout(F.relu(self.bn1(self.L1(x))))
        x = self.dropout(F.relu(self.bn2(self.L2(x))))
        return self.L3(x)
