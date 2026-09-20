import os

import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import k_hop_subgraph

from model import CoraGAT


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cora GAT Dashboard",
    page_icon="🧠",
    layout="wide"
)

HIDDEN_FEATURES = 32
NUM_HEADS = 4

MODEL_PATH = "gat_cora_best_model.pt"

SEED = 42

CLASS_NAMES = [
    "Case Based",
    "Genetic Algorithms",
    "Neural Networks",
    "Probabilistic Methods",
    "Reinforcement Learning",
    "Rule Learning",
    "Theory"
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_data():

    dataset = Planetoid(
        root="/tmp/Cora",
        name="Cora"
    )

    data = dataset[0]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    data = data.to(device)

    return dataset, data, device


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(
    num_features,
    num_classes,
    device
):

    model = CoraGAT(
        num_features,
        HIDDEN_FEATURES,
        num_classes,
        NUM_HEADS
    ).to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    return model


# ============================================================
# LOAD EVERYTHING
# ============================================================

dataset, data, device = load_data()

if not os.path.exists(MODEL_PATH):

    st.error(
        f"Model checkpoint not found: {MODEL_PATH}"
    )

    st.stop()


model = load_model(
    dataset.num_features,
    dataset.num_classes,
    device
)


# ============================================================
# MODEL OUTPUTS
# ============================================================

@st.cache_data
def get_model_outputs():

    with torch.no_grad():

        out, embeddings, att_edge_index, alpha = model(
            data.x,
            data.edge_index,
            return_att=True,
            return_embedding=True
        )

    probabilities = torch.softmax(
        out,
        dim=1
    )

    predictions = out.argmax(
        dim=1
    )

    return (
        out.cpu(),
        probabilities.cpu(),
        predictions.cpu(),
        embeddings.cpu(),
        att_edge_index.cpu(),
        alpha.cpu()
    )


(
    logits,
    probabilities,
    predictions,
    embeddings,
    att_edge_index,
    alpha
) = get_model_outputs()


# ============================================================
# GLOBAL METRICS
# ============================================================

def calculate_metrics(mask):

    y_true = data.y[mask].cpu().numpy()
    y_pred = predictions[mask].numpy()

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    }


train_metrics = calculate_metrics(
    data.train_mask
)

val_metrics = calculate_metrics(
    data.val_mask
)

test_metrics = calculate_metrics(
    data.test_mask
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Model")

st.sidebar.write(
    f"**Device:** `{device}`"
)

st.sidebar.write(
    f"**Nodes:** `{data.num_nodes}`"
)

st.sidebar.write(
    f"**Edges:** `{data.edge_index.shape[1]}`"
)

st.sidebar.write(
    f"**Features:** `{dataset.num_features}`"
)

st.sidebar.write(
    f"**Classes:** `{dataset.num_classes}`"
)

st.sidebar.write(
    f"**GAT heads:** `{NUM_HEADS}`"
)

st.sidebar.write(
    f"**Hidden features:** `{HIDDEN_FEATURES}`"
)

st.sidebar.divider()

st.sidebar.info(
    "The dashboard uses the best model saved by main.py."
)


# ============================================================
# TITLE
# ============================================================

st.title("🧠 Cora GAT Dashboard")

st.markdown(
    """
    Interactive visualization and analysis of a
    **Graph Attention Network (GAT)** trained on the
    **Cora citation network**.
    """
)


# ============================================================
# OVERVIEW METRICS
# ============================================================

st.header("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Test Accuracy",
        f"{test_metrics['accuracy'] * 100:.2f}%"
    )

with col2:
    st.metric(
        "Test Macro-F1",
        f"{test_metrics['f1'] * 100:.2f}%"
    )

with col3:
    st.metric(
        "Test Precision",
        f"{test_metrics['precision'] * 100:.2f}%"
    )

with col4:
    st.metric(
        "Test Recall",
        f"{test_metrics['recall'] * 100:.2f}%"
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

st.header("📚 Dataset")

dataset_col1, dataset_col2, dataset_col3 = st.columns(3)

with dataset_col1:
    st.metric(
        "Nodes",
        data.num_nodes
    )

with dataset_col2:
    st.metric(
        "Edges",
        data.edge_index.shape[1]
    )

with dataset_col3:
    st.metric(
        "Input Features",
        dataset.num_features
    )


st.write(
    "Cora contains scientific papers represented as nodes "
    "and citation relationships represented as edges."
)


# ============================================================
# CLASSIFICATION RESULTS
# ============================================================

st.header("🎯 Classification Results")

test_true = data.y[data.test_mask].cpu().numpy()
test_pred = predictions[data.test_mask].numpy()

per_class_f1 = f1_score(
    test_true,
    test_pred,
    average=None,
    zero_division=0
)

class_table = pd.DataFrame({
    "Class": CLASS_NAMES,
    "F1 Score": [
        f"{score * 100:.2f}%"
        for score in per_class_f1
    ]
})

st.dataframe(
    class_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.header("🔲 Confusion Matrix")

cm = confusion_matrix(
    test_true,
    test_pred
)

fig, ax = plt.subplots(
    figsize=(10, 8)
)

im = ax.imshow(cm)

ax.set_xticks(
    range(len(CLASS_NAMES))
)

ax.set_yticks(
    range(len(CLASS_NAMES))
)

ax.set_xticklabels(
    CLASS_NAMES,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(
    CLASS_NAMES
)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")

ax.set_title(
    "Cora GAT - Test Confusion Matrix"
)

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# ============================================================
# NODE EXPLORER
# ============================================================

st.header("🔍 Node Explorer")

node_col1, node_col2, node_col3 = st.columns(3)

with node_col1:

    target_node = st.number_input(
        "Node ID",
        min_value=0,
        max_value=data.num_nodes - 1,
        value=0,
        step=1
    )

with node_col2:

    attention_options = [
        "Mean Attention"
    ] + [
        f"Head {i + 1}"
        for i in range(NUM_HEADS)
    ]

    selected_attention = st.selectbox(
        "Attention",
        attention_options
    )

with node_col3:

    num_hops = st.selectbox(
        "Neighborhood",
        [1, 2, 3],
        index=0
    )


# ============================================================
# NODE PREDICTION
# ============================================================

node_prediction = int(
    predictions[target_node]
)

node_probability = float(
    probabilities[target_node, node_prediction]
)

node_true = int(
    data.y[target_node]
)

prediction_col1, prediction_col2, prediction_col3 = st.columns(3)

with prediction_col1:

    st.metric(
        "Predicted Class",
        CLASS_NAMES[node_prediction]
    )

with prediction_col2:

    st.metric(
        "True Class",
        CLASS_NAMES[node_true]
    )

with prediction_col3:

    st.metric(
        "Confidence",
        f"{node_probability * 100:.2f}%"
    )


if node_prediction == node_true:

    st.success(
        "✓ The model classified this node correctly."
    )

else:

    st.error(
        "✗ The model classified this node incorrectly."
    )


# ============================================================
# NODE PROBABILITIES
# ============================================================

st.subheader("Class Probabilities")

node_probs = probabilities[target_node].numpy()

probability_table = pd.DataFrame({
    "Class": CLASS_NAMES,
    "Probability": [
        f"{p * 100:.2f}%"
        for p in node_probs
    ]
})

st.dataframe(
    probability_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ATTENTION GRAPH
# ============================================================

st.subheader("👁 Attention Graph")

if alpha.dim() == 2:

    if selected_attention == "Mean Attention":

        selected_alpha = alpha.mean(
            dim=1
        )

    else:

        head_index = int(
            selected_attention.split()[-1]
        ) - 1

        selected_alpha = alpha[:, head_index]

else:

    selected_alpha = alpha


subset_nodes, subset_edges, mapping, edge_mask = k_hop_subgraph(
    node_idx=target_node,
    num_hops=num_hops,
    edge_index=att_edge_index,
    relabel_nodes=False
)

sub_alpha = selected_alpha[
    edge_mask
].numpy()

sub_edges = subset_edges.numpy()


G = nx.DiGraph()

for i in range(
    sub_edges.shape[1]
):

    src = int(
        sub_edges[0, i]
    )

    dst = int(
        sub_edges[1, i]
    )

    weight = float(
        sub_alpha[i]
    )

    G.add_edge(
        src,
        dst,
        weight=weight
    )


pos = nx.spring_layout(
    G,
    seed=SEED
)

fig, ax = plt.subplots(
    figsize=(11, 8)
)

node_colors = [
    "red" if node == target_node else "skyblue"
    for node in G.nodes()
]

nx.draw_networkx_nodes(
    G,
    pos,
    node_color=node_colors,
    node_size=750,
    ax=ax
)

nx.draw_networkx_labels(
    G,
    pos,
    font_weight="bold",
    ax=ax
)

edge_weights = [
    G[u][v]["weight"]
    for u, v in G.edges()
]

edge_widths = [
    max(1.0, weight * 10)
    for weight in edge_weights
]

nx.draw_networkx_edges(
    G,
    pos,
    width=edge_widths,
    edge_color=edge_weights,
    edge_cmap=plt.cm.Blues,
    arrows=True,
    arrowsize=15,
    ax=ax
)

edge_labels = {
    (u, v): f"{d['weight']:.2f}"
    for u, v, d in G.edges(
        data=True
    )
}

nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    font_size=8,
    ax=ax
)

ax.set_title(
    f"{selected_attention} - "
    f"Node {target_node} - "
    f"{num_hops}-hop Neighborhood"
)

ax.axis("off")

plt.tight_layout()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# ============================================================
# t-SNE
# ============================================================

st.header("🧠 Learned Node Embeddings")

st.write(
    "The following visualization uses the hidden representation "
    "produced by the first GAT layer rather than the final class logits."
)


@st.cache_data
def calculate_tsne(
    embedding_array
):

    tsne = TSNE(
        n_components=2,
        random_state=SEED,
        perplexity=30,
        init="pca",
        learning_rate="auto"
    )

    return tsne.fit_transform(
        embedding_array
    )


if st.button(
    "Generate t-SNE"
):

    with st.spinner(
        "Computing t-SNE..."
    ):

        embeddings_2d = calculate_tsne(
            embeddings.numpy()
        )

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    labels = data.y.cpu().numpy()

    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap="tab10",
        s=20,
        alpha=0.7
    )

    ax.set_title(
        "t-SNE of Learned GAT Embeddings"
    )

    ax.set_xlabel(
        "t-SNE Dimension 1"
    )

    ax.set_ylabel(
        "t-SNE Dimension 2"
    )

    legend_handles = []

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        handle = plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            label=class_name,
            color=plt.cm.tab10(
                class_id
            )
        )

        legend_handles.append(
            handle
        )

    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# ATTENTION DISTRIBUTION
# ============================================================

st.header("📈 Attention Distribution")

if alpha.dim() == 2:

    distribution_col1, distribution_col2 = st.columns(2)

    with distribution_col1:

        distribution_head = st.selectbox(
            "Select attention head",
            list(range(NUM_HEADS)),
            format_func=lambda x: f"Head {x + 1}"
        )

    with distribution_col2:

        st.metric(
            "Mean Attention",
            f"{alpha[:, distribution_head].mean():.4f}"
        )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.hist(
        alpha[:, distribution_head].numpy(),
        bins=50,
        alpha=0.8
    )

    ax.set_xlabel(
        "Attention Coefficient"
    )

    ax.set_ylabel(
        "Number of Edges"
    )

    ax.set_title(
        f"Attention Distribution - "
        f"Head {distribution_head + 1}"
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Cora GAT — PyTorch Geometric + Streamlit"
)