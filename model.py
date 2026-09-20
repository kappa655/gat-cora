import copy
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from torch_geometric.datasets import Planetoid

from model import CoraGAT


# ============================================================
# PARAMETERS
# ============================================================

HIDDEN_FEATURES = 32
NUM_HEADS = 4

LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

MAX_EPOCHS = 1000
PATIENCE = 50

SEED = 42

CHECKPOINT_PATH = "gat_cora_best_model.pt"


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# LOAD DATA
# ============================================================

dataset = Planetoid(root="/tmp/Cora", name="Cora")
data = dataset[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)


# ============================================================
# MODEL
# ============================================================

model = CoraGAT(
    dataset.num_features,
    HIDDEN_FEATURES,
    dataset.num_classes,
    NUM_HEADS
).to(device)

print("=" * 70)
print("Cora GAT Training")
print("=" * 70)
print(f"Device:          {device}")
print(f"Input features:  {dataset.num_features}")
print(f"Hidden features: {HIDDEN_FEATURES}")
print(f"Attention heads: {NUM_HEADS}")
print(f"Classes:         {dataset.num_classes}")
print(f"Max epochs:      {MAX_EPOCHS}")
print(f"Early stopping:  {PATIENCE}")
print("=" * 70)


# ============================================================
# LOSS & OPTIMIZER
# ============================================================

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, logits):
    predictions = logits.argmax(dim=1)

    y_true = y_true.cpu().numpy()
    predictions = predictions.cpu().numpy()

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, average="macro", zero_division=0),
        "recall": recall_score(y_true, predictions, average="macro", zero_division=0),
        "f1": f1_score(y_true, predictions, average="macro", zero_division=0)
    }


# ============================================================
# TRAIN
# ============================================================

def train():
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)

    loss = criterion(
        out[data.train_mask],
        data.y[data.train_mask]
    )

    loss.backward()
    optimizer.step()

    return loss.item()


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate():
    model.eval()

    out = model(data.x, data.edge_index)

    results = {}

    for split, mask in [
        ("train", data.train_mask),
        ("val", data.val_mask),
        ("test", data.test_mask)
    ]:
        loss = criterion(out[mask], data.y[mask]).item()
        metrics = calculate_metrics(data.y[mask], out[mask])

        results[split] = {
            "loss": loss,
            **metrics
        }

    return results, out


# ============================================================
# TRAINING HISTORY
# ============================================================

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "train_precision": [],
    "val_precision": [],
    "train_recall": [],
    "val_recall": [],
    "train_f1": [],
    "val_f1": []
}


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_f1 = -1.0
best_epoch = 0
patience_counter = 0

print("\nStarting training...\n")

for epoch in range(1, MAX_EPOCHS + 1):
    train_loss = train()
    results, _ = evaluate()

    train_metrics = results["train"]
    val_metrics = results["val"]

    # Save training history
    history["train_loss"].append(train_metrics["loss"])
    history["val_loss"].append(val_metrics["loss"])

    history["train_acc"].append(train_metrics["accuracy"])
    history["val_acc"].append(val_metrics["accuracy"])

    history["train_precision"].append(train_metrics["precision"])
    history["val_precision"].append(val_metrics["precision"])

    history["train_recall"].append(train_metrics["recall"])
    history["val_recall"].append(val_metrics["recall"])

    history["train_f1"].append(train_metrics["f1"])
    history["val_f1"].append(val_metrics["f1"])

    # Save best model based on validation Macro-F1
    if val_metrics["f1"] > best_val_f1:
        best_val_f1 = val_metrics["f1"]
        best_epoch = epoch
        patience_counter = 0

        torch.save(model.state_dict(), CHECKPOINT_PATH)

    else:
        patience_counter += 1

    # Print progress
    if epoch == 1 or epoch % 20 == 0:
        print(
            f"Epoch {epoch:04d} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_metrics['accuracy'] * 100:.2f}% | "
            f"Val Acc: {val_metrics['accuracy'] * 100:.2f}% | "
            f"Val F1: {val_metrics['f1'] * 100:.2f}% | "
            f"Best F1: {best_val_f1 * 100:.2f}%"
        )

    # Early stopping
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        print(f"Best epoch: {best_epoch}")
        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )
)

model.eval()


# ============================================================
# FINAL EVALUATION
# ============================================================

results, final_out = evaluate()

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

for split in ["train", "val", "test"]:
    metrics = results[split]

    print(f"\n{split.upper()}")
    print(f"Loss:      {metrics['loss']:.4f}")
    print(f"Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"Macro F1:  {metrics['f1'] * 100:.2f}%")

print("\n" + "=" * 70)
print(f"Best Validation Macro-F1: {best_val_f1 * 100:.2f}%")
print(f"Best Epoch: {best_epoch}")
print(f"Model saved to: {CHECKPOINT_PATH}")
print("=" * 70)


# ============================================================
# PER-CLASS F1
# ============================================================

predictions = final_out.argmax(dim=1)

test_true = data.y[data.test_mask].cpu().numpy()
test_pred = predictions[data.test_mask].cpu().numpy()

class_names = [
    "Theory",
    "Reinforcement Learning",
    "Genetic Algorithms",
    "Neural Networks",
    "Probabilistic Methods",
    "Case Based",
    "Rule Learning"
]

per_class_f1 = f1_score(
    test_true,
    test_pred,
    average=None,
    zero_division=0
)

print("\nPER-CLASS F1")
print("-" * 50)

for i, score in enumerate(per_class_f1):
    print(f"{class_names[i]:25s}: {score * 100:.2f}%")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(test_true, test_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(10, 8))

disp.plot(
    ax=ax,
    xticks_rotation=45,
    cmap="Blues",
    colorbar=False
)

plt.title(
    "Cora GAT - Test Confusion Matrix",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()
plt.savefig("cora_gat_confusion_matrix.png", dpi=300)
plt.show()


# ============================================================
# LOSS CURVE
# ============================================================

epochs = range(1, len(history["train_loss"]) + 1)

plt.figure(figsize=(10, 6))

plt.plot(epochs, history["train_loss"], label="Train Loss")
plt.plot(epochs, history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Cora GAT - Training and Validation Loss")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("cora_gat_loss_curve.png", dpi=300)
plt.show()


# ============================================================
# ACCURACY CURVE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(epochs, history["train_acc"], label="Train Accuracy")
plt.plot(epochs, history["val_acc"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Cora GAT - Accuracy")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("cora_gat_accuracy_curve.png", dpi=300)
plt.show()


# ============================================================
# MACRO-F1 CURVE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(epochs, history["train_f1"], label="Train Macro-F1")
plt.plot(epochs, history["val_f1"], label="Validation Macro-F1")

plt.xlabel("Epoch")
plt.ylabel("Macro-F1")
plt.title("Cora GAT - Macro-F1")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("cora_gat_f1_curve.png", dpi=300)
plt.show()
