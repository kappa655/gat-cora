import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import k_hop_subgraph

from model import CoraGAT


# ============================================================
# PARAMETERS
# ============================================================

HIDDEN_FEATURES = 32
NUM_HEADS = 4

MODEL_PATH = "gat_cora_best_model.pt"

TARGET_NODES = [0, 100, 500]

NUM_HOPS = 1


# ============================================================
# LOAD DATA
# ============================================================

dataset = Planetoid(root="/tmp/Cora", name="Cora")
data = dataset[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)


# ============================================================
# LOAD MODEL
# ============================================================

model = CoraGAT(
    dataset.num_features,
    HIDDEN_FEATURES,
    dataset.num_classes,
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

print("=" * 70)
print("Cora GAT Visualization")
print("=" * 70)
print(f"Device: {device}")
print(f"Model:  {MODEL_PATH}")
print("=" * 70)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
    "Theory",
    "Reinforcement Learning",
    "Genetic Algorithms",
    "Neural Networks",
    "Probabilistic Methods",
    "Case Based",
    "Rule Learning"
]


# ============================================================
# A. t-SNE OF HIDDEN EMBEDDINGS
# ============================================================

def plot_tsne():

    print("\n[1/3] Computing hidden GAT embeddings...")

    with torch.no_grad():
        out, embeddings = model(
            data.x,
            data.edge_index,
            return_embedding=True
        )

    embeddings = embeddings.cpu().numpy()
    labels = data.y.cpu().numpy()

    print(f"Embedding shape: {embeddings.shape}")
    print("Running t-SNE...")

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,
        init="pca",
        learning_rate="auto"
    )

    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(11, 9))

    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.75,
        s=25
    )

    cbar = plt.colorbar(
        scatter,
        ticks=range(len(class_names))
    )

    cbar.ax.set_yticklabels(class_names)

    plt.title(
        "t-SNE of Learned GAT Node Embeddings",
        fontsize=15,
        fontweight="bold"
    )

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")

    plt.grid(
        True,
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "cora_gat_tsne_embeddings.png",
        dpi=300
    )

    plt.show()

    print("Saved: cora_gat_tsne_embeddings.png")


# ============================================================
# B. ATTENTION SUBGRAPH
# ============================================================

def plot_attention_subgraph(target_node=0, num_hops=1, head=None):

    print(
        f"\nAttention visualization for node {target_node}"
    )

    with torch.no_grad():
        out, att_edge_index, alpha = model(
            data.x,
            data.edge_index,
            return_att=True
        )

    # alpha shape:
    # [number_of_edges, number_of_heads]

    if alpha.dim() == 2:

        if head is None:
            selected_alpha = alpha.mean(dim=1)
            attention_title = "Mean Attention"

        else:
            selected_alpha = alpha[:, head]
            attention_title = f"Attention Head {head + 1}"

    else:
        selected_alpha = alpha
        attention_title = "Attention"

    # --------------------------------------------------------
    # Extract k-hop subgraph
    # --------------------------------------------------------

    subset_nodes, subset_edges, mapping, edge_mask = k_hop_subgraph(
        node_idx=target_node,
        num_hops=num_hops,
        edge_index=att_edge_index,
        relabel_nodes=False
    )

    sub_alpha = selected_alpha[edge_mask].cpu().numpy()
    sub_edges = subset_edges.cpu().numpy()

    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------

    G = nx.DiGraph()

    for i in range(sub_edges.shape[1]):
        src = int(sub_edges[0, i])
        dst = int(sub_edges[1, i])
        weight = float(sub_alpha[i])

        G.add_edge(
            src,
            dst,
            weight=weight
        )

    pos = nx.spring_layout(
        G,
        seed=42
    )

    weights = [
        G[u][v]["weight"]
        for u, v in G.edges()
    ]

    # --------------------------------------------------------
    # Plot nodes
    # --------------------------------------------------------

    plt.figure(figsize=(10, 8))

    node_colors = [
        "red" if node == target_node else "skyblue"
        for node in G.nodes()
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=650
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_color="black",
        font_weight="bold"
    )

    # --------------------------------------------------------
    # Plot edges
    # --------------------------------------------------------

    edge_widths = [
        max(1.0, weight * 10)
        for weight in weights
    ]

    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        edge_color=weights,
        edge_cmap=plt.cm.Blues,
        arrows=True,
        arrowsize=15
    )

    # --------------------------------------------------------
    # Attention labels
    # --------------------------------------------------------

    edge_labels = {
        (u, v): f"{d['weight']:.2f}"
        for u, v, d in G.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=8
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    plt.title(
        f"Cora GAT - {attention_title}\n"
        f"Node {target_node}, {num_hops}-hop neighborhood",
        fontsize=13,
        fontweight="bold"
    )

    plt.axis("off")
    plt.tight_layout()

    if head is None:
        filename = f"cora_gat_attention_node_{target_node}_mean.png"
    else:
        filename = f"cora_gat_attention_node_{target_node}_head_{head + 1}.png"

    plt.savefig(
        filename,
        dpi=300
    )

    plt.show()

    print(f"Saved: {filename}")


# ============================================================
# C. ATTENTION DISTRIBUTION
# ============================================================

def plot_attention_distribution():

    print("\n[3/3] Computing attention distributions...")

    with torch.no_grad():
        out, att_edge_index, alpha = model(
            data.x,
            data.edge_index,
            return_att=True
        )

    alpha_np = alpha.cpu().numpy()

    if alpha_np.ndim != 2:
        print("Attention weights do not contain multiple heads.")
        return

    num_heads = alpha_np.shape[1]

    for head in range(num_heads):

        plt.figure(figsize=(9, 6))

        plt.hist(
            alpha_np[:, head],
            bins=50,
            alpha=0.8
        )

        plt.xlabel("Attention coefficient")
        plt.ylabel("Number of edges")

        plt.title(
            f"GAT Attention Distribution - Head {head + 1}"
        )

        plt.grid(alpha=0.3)

        plt.tight_layout()

        filename = (
            f"cora_gat_attention_distribution_head_{head + 1}.png"
        )

        plt.savefig(
            filename,
            dpi=300
        )

        plt.show()

        print(f"Saved: {filename}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # 1. t-SNE of learned embeddings
    plot_tsne()

    # 2. Mean attention for selected nodes
    for node in TARGET_NODES:
        plot_attention_subgraph(
            target_node=node,
            num_hops=NUM_HOPS,
            head=None
        )

    # 3. Attention of every head for node 0
    for head in range(NUM_HEADS):
        plot_attention_subgraph(
            target_node=0,
            num_hops=NUM_HOPS,
            head=head
        )

    # 4. Attention distributions
    plot_attention_distribution()

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)
