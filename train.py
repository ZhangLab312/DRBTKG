import copy
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader






seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "data").is_dir() else SCRIPT_DIR.parent





node_type1 = "drug"
node_type2 = "disease"
rel = "indication"

EDGE_TYPE = (node_type1, rel, node_type2)
REV_EDGE_TYPE = (node_type2, "rev_indication", node_type1)

config = {
    "num_samples": 512,
    "batch_size": 164,
    "dropout": 0.2,
    "epochs": 300,
    "semantic_embedding_dim": 64,
    "projected_embedding_dim": 64,
    "use_semantic_projection": True,

    "semantic_embedding_path": str(
        PROJECT_ROOT / "data" / "drug_text_embeddings"
        / "drug_text_embeddings_64d.pt"
    ),
    "semantic_mapping_path": str(
        PROJECT_ROOT / "data" / "drug_text_embeddings"
        / "drug_text_mappings.pkl"
    ),
}





primekg_file = str(PROJECT_ROOT / "data" / "kg.csv")
df = pd.read_csv(
    primekg_file,
    sep=",",
    dtype={"x_index": str, "y_index": str, "x_id": str, "y_id": str},
)



drug_index_to_drugbank_id = {}
for side in ("x", "y"):
    for _, row in df.loc[
        df[f"{side}_type"] == "drug",
        [f"{side}_index", f"{side}_id"],
    ].dropna().iterrows():
        drug_index_to_drugbank_id.setdefault(
            str(row[f"{side}_index"]),
            str(row[f"{side}_id"]),
        )

drug_disease_pairs = df[df["relation"] == rel]
drugs, diseases = [], []

for _, row in drug_disease_pairs.iterrows():
    if row["x_type"] == node_type1:
        drugs.append(row["x_index"])
    if row["x_type"] == node_type2:
        diseases.append(row["x_index"])

    if row["y_type"] == node_type1:
        drugs.append(row["y_index"])
    if row["y_type"] == node_type2:
        diseases.append(row["y_index"])

drugs = list(set(drugs))
diseases = list(set(diseases))

new_df = pd.DataFrame()
new_df[0] = df["x_type"] + "::" + df["x_index"].astype(str)
new_df[1] = df["relation"]
new_df[2] = df["y_type"] + "::" + df["y_index"].astype(str)

df = new_df.drop_duplicates()
triplets = df.values.tolist()

entity_dictionary = {}


def insert_entry(entry, ent_type, dic):
    if ent_type not in dic:
        dic[ent_type] = {}

    if entry not in dic[ent_type]:
        dic[ent_type][entry] = len(dic[ent_type])

    return dic


for triple in triplets:
    src = triple[0]
    dst = triple[2]

    src_type = src.split("::")[0]
    dst_type = dst.split("::")[0]

    insert_entry(src, src_type, entity_dictionary)
    insert_entry(dst, dst_type, entity_dictionary)


edge_dictionary = {}

for triple in triplets:
    src = triple[0]
    dst = triple[2]

    src_type = src.split("::")[0]
    dst_type = dst.split("::")[0]

    src_int_id = entity_dictionary[src_type][src]
    dst_int_id = entity_dictionary[dst_type][dst]

    edge_type = (src_type, triple[1], dst_type)

    if edge_type not in edge_dictionary:
        edge_dictionary[edge_type] = []

    edge_dictionary[edge_type].append((src_int_id, dst_int_id))


data = HeteroData()
node_feature_dims = {}
base_feature_dims = {}

for i, node_type in enumerate(entity_dictionary.keys()):
    if node_type != "drug":
        base_feature_dim = 768
        data[node_type].x = torch.ones(
            (len(entity_dictionary[node_type]), base_feature_dim)
        ) * i
    else:
        base_feature_dim = 767
        data[node_type].x = torch.rand(
            (len(entity_dictionary[node_type]), base_feature_dim)
        )

    base_feature_dims[node_type] = base_feature_dim

    if config["use_semantic_projection"] and node_type in {"drug", "disease"}:
        node_feature_dims[node_type] = (
            base_feature_dim + config["projected_embedding_dim"]
        )
    else:
        node_feature_dims[node_type] = base_feature_dim

    data[node_type].id = torch.arange(
        len(entity_dictionary[node_type]),
        dtype=torch.long,
    )

    entity_names = [""] * len(entity_dictionary[node_type])

    for entity_name, local_id in entity_dictionary[node_type].items():
        entity_names[local_id] = entity_name.split("::", 1)[-1]

        if node_type == "drug":
            entity_names[local_id] = drug_index_to_drugbank_id.get(
                entity_names[local_id],
                entity_names[local_id],
            )

    data[node_type].entity_names = entity_names


for edge_type, edges in edge_dictionary.items():
    data[edge_type].edge_index = (
        torch.tensor(edges, dtype=torch.long)
        .t()
        .contiguous()
    )






try:
    with open(PROJECT_ROOT / "data" / "train_data" / "train1.pkl", "rb") as file:
        train_data = pickle.load(file)

    with open(PROJECT_ROOT / "data" / "train_data" / "test1.pkl", "rb") as file:
        val_data = pickle.load(file)

    print("Training and validation splits loaded successfully.")

except Exception as e:
    raise RuntimeError(
        "Failed to load train1.pkl or test1.pkl. "
        "The full graph must never be used as a fallback."
    ) from e





from model import (
    EnhancedMLPPredictor,
    FixedSemanticEmbeddingManager,
    ProjectionEnhancedHGAT,
    SemanticProjectionNetwork,
    configure_model_context,
)

configure_model_context(
    device=device,
    node_type1=node_type1,
    node_type2=node_type2,
)
semantic_manager = FixedSemanticEmbeddingManager(config)







def attach_drug_text_embeddings(graph):
    num_drugs = graph["drug"].num_nodes
    aligned = torch.zeros(
        num_drugs,
        config["semantic_embedding_dim"],
        dtype=torch.float,
    )
    matched = 0

    for node_id in range(num_drugs):
        if node_id >= len(data["drug"].entity_names):
            continue

        vector = semantic_manager.get_semantic_embedding(
            data["drug"].entity_names[node_id]
        )
        if vector is not None:
            aligned[node_id] = vector.cpu()
            matched += 1

    graph["drug"].text_embedding = aligned
    print(f"Pre-aligned drug text embeddings: {matched}/{num_drugs}")


attach_drug_text_embeddings(train_data)
attach_drug_text_embeddings(val_data)





def is_excluded_target_edge_type(edge_type):
    src_type, relation_name, dst_type = edge_type
    return (
        {src_type, dst_type} == {node_type1, node_type2}
        and relation_name.removeprefix("rev_")
        in {"indication", "contraindication"}
    )


def get_target_edge_types(graph):
    return [
        edge_type
        for edge_type in graph.edge_types
        if is_excluded_target_edge_type(edge_type)
    ]


def remove_all_target_edges(graph):
    graph = copy.deepcopy(graph)
    target_types = get_target_edge_types(graph)

    if EDGE_TYPE not in target_types:
        raise RuntimeError(
            f"Missing target relation {EDGE_TYPE}. "
            "Check relation names in kg.csv and split files."
        )

    for edge_type in target_types:
        del graph[edge_type]

    remaining_target_types = get_target_edge_types(graph)

    if remaining_target_types:
        raise RuntimeError(
            "Target relations remain in the message-passing graph: "
            f"{remaining_target_types}"
        )

    return graph, target_types


def prepare_train_graph_and_labels(graph):
    train_positive_edge_index = graph[EDGE_TYPE].edge_index.cpu()

    train_edge_label = torch.ones(
        train_positive_edge_index.size(1),
        dtype=torch.float,
    )

    train_graph, removed_types = remove_all_target_edges(graph)

    print("Removed target relations from training graph:")

    for edge_type in removed_types:
        print(f"  {edge_type}")

    return train_graph, train_positive_edge_index, train_edge_label


def prepare_val_labels(graph):
    if (
        hasattr(graph[EDGE_TYPE], "edge_label_index")
        and graph[EDGE_TYPE].edge_label_index is not None
    ):
        val_edge_label_index = graph[EDGE_TYPE].edge_label_index.cpu()

        if (
            hasattr(graph[EDGE_TYPE], "edge_label")
            and graph[EDGE_TYPE].edge_label is not None
        ):
            val_edge_label = graph[EDGE_TYPE].edge_label.float().cpu()
        else:
            val_edge_label = torch.ones(
                val_edge_label_index.size(1),
                dtype=torch.float,
            )
    else:
        val_edge_label_index = graph[EDGE_TYPE].edge_index.cpu()
        val_edge_label = torch.ones(
            val_edge_label_index.size(1),
            dtype=torch.float,
        )

    return val_edge_label_index, val_edge_label







def define_model(message_passing_graph, dropout):
    projection_net = None

    if (
        config["use_semantic_projection"]
        and semantic_manager.semantic_embeddings is not None
    ):
        projection_net = SemanticProjectionNetwork(
            semantic_embedding_dim=config["semantic_embedding_dim"],
            projected_embedding_dim=config["projected_embedding_dim"],
            dropout=dropout,
        ).to(device)

    gat = ProjectionEnhancedHGAT(
        node_feature_dims=node_feature_dims,
        hidden_channels=[64, 64, 64, 64],
        out_channels=64,
        num_heads=[8, 8, 8],
        num_layers=3,
        dropout=dropout,
        message_passing_graph=message_passing_graph,
        projected_embedding_dim=config["projected_embedding_dim"],
        use_semantic_projection=config["use_semantic_projection"],
    ).to(device)

    predictor = EnhancedMLPPredictor(64, dropout).to(device)

    return projection_net, gat, predictor





def get_num_neighbors(graph, num_hops=3):
    return {
        edge_type: [config["num_samples"]] * num_hops
        for edge_type in graph.edge_types
    }


def define_loaders():
    train_graph, train_edge_label_index, train_edge_label = (
        prepare_train_graph_and_labels(train_data)
    )

    val_edge_label_index, val_edge_label = prepare_val_labels(val_data)


    val_graph = copy.deepcopy(train_graph)

    for graph in [train_graph, val_graph]:
        assert EDGE_TYPE not in graph.edge_types
        assert REV_EDGE_TYPE not in graph.edge_types

    kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = LinkNeighborLoader(
        data=train_graph,
        num_neighbors=get_num_neighbors(train_graph, num_hops=3),
        edge_label_index=(EDGE_TYPE, train_edge_label_index),
        edge_label=train_edge_label,
        neg_sampling_ratio=1.0,
        shuffle=True,
        **kwargs,
    )

    val_loader_kwargs = {
        "data": val_graph,
        "num_neighbors": get_num_neighbors(val_graph, num_hops=3),
        "edge_label_index": (EDGE_TYPE, val_edge_label_index),
        "edge_label": val_edge_label,
        "shuffle": False,
        **kwargs,
    }

    if not (val_edge_label == 0).any():
        val_loader_kwargs["neg_sampling_ratio"] = 1.0

    val_loader = LinkNeighborLoader(**val_loader_kwargs)

    return train_loader, val_loader, train_graph





def compute_loss(scores, labels):
    pos_count = max((labels == 1).sum().item(), 1)
    neg_count = max((labels == 0).sum().item(), 1)

    sample_weights = torch.ones_like(
        labels,
        dtype=torch.float,
        device=labels.device,
    )

    sample_weights[labels == 1] = neg_count / (pos_count + neg_count)
    sample_weights[labels == 0] = pos_count / (pos_count + neg_count)

    return F.binary_cross_entropy_with_logits(
        scores,
        labels,
        weight=sample_weights,
    )


def compute_metrics(predictions, labels):
    probabilities = torch.sigmoid(predictions).detach().cpu().numpy()
    predicted_labels = (probabilities > 0.5).astype(int)
    true_labels = labels.detach().cpu().numpy()

    return {
        "auroc": roc_auc_score(true_labels, probabilities),
        "aupr": average_precision_score(true_labels, probabilities),
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(
            true_labels,
            predicted_labels,
            zero_division=0,
        ),
        "recall": recall_score(
            true_labels,
            predicted_labels,
            zero_division=0,
        ),
        "f1": f1_score(
            true_labels,
            predicted_labels,
            zero_division=0,
        ),
    }


def assert_no_target_edges(batch):
    target_types = [
        edge_type
        for edge_type in batch.edge_index_dict
        if is_excluded_target_edge_type(edge_type)
    ]
    if target_types:
        raise RuntimeError(
            "indication or contraindication edges entered message passing: "
            f"{target_types}"
        )


def train_epoch(projection_net, gat, predictor, loader, optimizer):
    gat.train()
    predictor.train()

    if projection_net is not None:
        projection_net.train()

    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)
        assert_no_target_edges(batch)

        optimizer.zero_grad()

        edge_label_index = batch[EDGE_TYPE].edge_label_index
        edge_label = batch[EDGE_TYPE].edge_label.float()

        drug_embeddings, disease_embeddings = gat(
            batch.x_dict,
            batch.edge_index_dict,
            batch,
            projection_net,
        )

        drug_embeddings = drug_embeddings[edge_label_index[0]]
        disease_embeddings = disease_embeddings[edge_label_index[1]]

        scores = predictor(drug_embeddings, disease_embeddings)[:, 0]

        loss = compute_loss(scores, edge_label)
        loss.backward()
        optimizer.step()

        total_examples += edge_label.numel()
        total_loss += loss.detach().item() * edge_label.numel()

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate(projection_net, gat, predictor, loader):
    gat.eval()
    predictor.eval()

    if projection_net is not None:
        projection_net.eval()

    all_scores = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        assert_no_target_edges(batch)

        edge_label_index = batch[EDGE_TYPE].edge_label_index
        edge_label = batch[EDGE_TYPE].edge_label.float()

        drug_embeddings, disease_embeddings = gat(
            batch.x_dict,
            batch.edge_index_dict,
            batch,
            projection_net,
        )

        drug_embeddings = drug_embeddings[edge_label_index[0]]
        disease_embeddings = disease_embeddings[edge_label_index[1]]

        scores = predictor(drug_embeddings, disease_embeddings)[:, 0]

        all_scores.append(scores)
        all_labels.append(edge_label)

    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return (
        compute_loss(scores, labels).item(),
        compute_metrics(scores, labels),
    )





def run(cfg):
    train_loader, val_loader, message_passing_graph = define_loaders()

    projection_net, gat, predictor = define_model(
        message_passing_graph,
        cfg["dropout"],
    )

    optimizer_parameters = [
        {"params": gat.parameters(), "lr": 0.001},
        {"params": predictor.parameters(), "lr": 0.0005},
    ]

    if projection_net is not None:
        optimizer_parameters.append(
            {"params": projection_net.parameters(), "lr": 0.001}
        )

    optimizer = torch.optim.AdamW(optimizer_parameters)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"],
        eta_min=0,
    )

    best_metrics = None
    best_epoch = -1

    for epoch in range(cfg["epochs"]):
        train_loss = train_epoch(
            projection_net,
            gat,
            predictor,
            train_loader,
            optimizer,
        )

        val_loss, metrics = evaluate(
            projection_net,
            gat,
            predictor,
            val_loader,
        )

        if best_metrics is None or metrics["auroc"] > best_metrics["auroc"]:
            best_metrics = metrics.copy()
            best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "gat_state_dict": gat.state_dict(),
                "predictor_state_dict": predictor.state_dict(),
                "config": cfg,
                "metrics": best_metrics,
            }

            if projection_net is not None:
                checkpoint["projection_net_state_dict"] = (
                    projection_net.state_dict()
                )

            torch.save(checkpoint, "best_model_no_target_edges.pth")

        if epoch % 10 == 0 or epoch == cfg["epochs"] - 1:
            print(
                f"Epoch: {epoch:03d}, "
                f"Loss: {train_loss:.4f}, "
                f"ValLoss: {val_loss:.4f}"
            )
            print(
                f"AUROC: {metrics['auroc']:.4f}, "
                f"AUPR: {metrics['aupr']:.4f}, "
                f"Accuracy: {metrics['accuracy']:.4f}, "
                f"Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, "
                f"F1-score: {metrics['f1']:.4f}"
            )

        scheduler.step()

    print(f"\nBest epoch: {best_epoch}")
    print(f"AUROC: {best_metrics['auroc']:.4f}")
    print(f"AUPR: {best_metrics['aupr']:.4f}")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"F1-score: {best_metrics['f1']:.4f}")


if __name__ == "__main__":
    run(config)
