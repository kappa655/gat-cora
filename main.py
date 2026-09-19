import torch
from torch_geometric.datasets import Planetoid
from model import CoraGAT

# Parameters
HIDDEN_FEATURES = 32
NUM_HEADS = 4

# Load Data
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CoraGAT(dataset.num_features, HIDDEN_FEATURES, dataset.num_classes, NUM_HEADS).to(device)
data = data.to(device)

print("Model initialized successfully on:", device)

# Training setup
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)  # Χρήση model(...) αντί για model.forward(...)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Testing
@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    accuracy = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = (pred[mask] == data.y[mask]).sum()
        ratio = float(correct) / float(mask.sum())
        accuracy.append(ratio)
    return accuracy

# Main Loop με σωστό Model Checkpointing
best_val_acc = 0.0

for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()
    
    # Σώζουμε το μοντέλο ΜΟΝΟ όταν βελτιώνεται το Validation Accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "gat_cora_best_model.pt")
        
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train: {train_acc*100:.1f}% | Val: {val_acc*100:.1f}% | Best Val: {best_val_acc*100:.1f}%")

print(f"\nTraining complete! Best Validation Accuracy: {best_val_acc*100:.2f}% saved to gat_cora_best_model.pt")