import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import k_hop_subgraph
from model import CoraGAT

# 1. Παράμετροι & Φόρτωση Δεδομένων
HIDDEN_FEATURES = 32
NUM_HEADS = 4

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# 2. Φόρτωση Μοντέλου μέσω state_dict
model = CoraGAT(dataset.num_features, HIDDEN_FEATURES, dataset.num_classes, NUM_HEADS).to(device)
model.load_state_dict(torch.load("gat_cora_best_model.pt", map_location=device, weights_only=True))
model.eval()

print("To μοντέλο φορτώθηκε επιτυχώς!")

# ---------------------------------------------------------
# A. t-SNE VISUALIZATION
# ---------------------------------------------------------
def plot_tsne():
    print("\n[1/2] Υπολογισμός t-SNE Embeddings...")
    with torch.no_grad():
        # Παίρνουμε τα raw outputs (logits) του μοντέλου
        out = model(data.x, data.edge_index)
        embeddings = out.cpu().numpy()
        labels = data.y.cpu().numpy()

    # Εφαρμογή t-SNE για μείωση διαστάσεων από 7 σε 2
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Σχεδιασμός
    plt.figure(figsize=(10, 8))
    class_names = ['Theory', 'Reinforcement Learning', 'Genetic Algorithms', 
                   'Neural Networks', 'Probabilistic Methods', 'Case Based', 'Rule Learning']
    
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=labels, 
        cmap='tab10', 
        alpha=0.8, 
        s=30
    )
    
    cbar = plt.colorbar(scatter, ticks=range(7))
    cbar.ax.set_yticklabels(class_names)
    
    plt.title("t-SNE Visualization of Cora Node Embeddings (GAT)", fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("cora_gat_tsne.png", dpi=300)
    plt.show()
    print("Το t-SNE plot αποθηκεύτηκε ως 'cora_gat_tsne.png'!")

# ---------------------------------------------------------
# B. ATTENTION WEIGHTS SUBGRAPH VISUALIZATION
# ---------------------------------------------------------
def plot_attention_subgraph(target_node=0, num_hops=1):
    print(f"\n[2/2] Υπολογισμός Attention Weights γύρω από το Node {target_node}...")
    
    with torch.no_grad():
        # Παίρνουμε τα logits και τα attention weights
        out, att_edge_index, alpha = model(data.x, data.edge_index, return_att=True)
        
        # Αν τα heads > 1, παίρνουμε τον μέσο όρο των attention scores των κεφαλών (heads)
        if alpha.dim() > 1 and alpha.size(1) > 1:
            alpha = alpha.mean(dim=1)

    # Εξαγωγή του υπο-γραφήματος γύρω από το target_node
    subset_nodes, subset_edges, mapping, edge_mask = k_hop_subgraph(
        node_idx=target_node,
        num_hops=num_hops,
        edge_index=att_edge_index,
        relabel_nodes=False
    )

    # Φιλτράρισμα των attention values για τα ακριβή edges του υπο-γραφήματος
    sub_alpha = alpha[edge_mask].cpu().numpy()
    sub_edges = subset_edges.cpu().numpy()

    # Δημιουργία NetworkX Graph
    G = nx.DiGraph()
    for i in range(sub_edges.shape[1]):
        src = sub_edges[0, i]
        dst = sub_edges[1, i]
        weight = sub_alpha[i]
        G.add_edge(src, dst, weight=weight)

    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    plt.figure(figsize=(9, 7))
    
    # Σχεδιασμός Nodes
    node_colors = ['red' if node == target_node else 'skyblue' for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600)
    nx.draw_networkx_labels(G, pos, font_color='black', font_weight='bold')

    # Σχεδιασμός Edges (το πάχος εξαρτάται από το Attention Score)
    edge_widths = [w * 10 for w in weights]
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=weights, 
                           edge_cmap=plt.cm.Blues, arrows=True, arrowsize=15)

    # Labeling των βαρών προσοχής (Attention scores) πάνω στις ακμές
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(f"GAT Attention Weights around Paper Node {target_node}", fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("cora_gat_attention.png", dpi=300)
    plt.show()
    print("Το Attention plot αποθηκεύτηκε ως 'cora_gat_attention.png'!")

# Εκτέλεση των Visualizations
if __name__ == "__main__":
    plot_tsne()
    plot_attention_subgraph(target_node=0, num_hops=1)