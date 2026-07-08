import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import matplotlib.pyplot as plt

from src.data.dataset import NEODataset, get_normalization_transform
from src.models.model import NEOModel

RUN_TEST = False

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

data_dir = Path.cwd() / "data"
parquet_file = data_dir / "neos_ml.parquet"

full_dataset = NEODataset(parquet_file)

train_size = int(0.7 * len(full_dataset))
cv_size = int((len(full_dataset) - train_size) / 2.0)
test_size = len(full_dataset) - train_size - cv_size

train_dataset, cv_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, cv_size, test_size],
    generator=torch.Generator().manual_seed(42),
)

train_labels = full_dataset.labels[train_dataset.indices]
num_pos = train_labels.sum()
num_neg = len(train_labels) - num_pos
pos_weight = (num_neg / num_pos).to(device)

print(f"hazardous in train: {num_pos.int()}/{len(train_labels)}  pos_weight={pos_weight:.1f}")


train_features = full_dataset.features[train_dataset.indices]
mean_vector = torch.nanmean(train_features, dim=0)
diff = train_features - mean_vector
std_vector = torch.sqrt(torch.nanmean(diff * diff, dim=0))
full_dataset.transform = get_normalization_transform(mean_vector, std_vector)

train_loader_nn = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
cv_loader_nn = DataLoader(cv_dataset, batch_size=32, shuffle=False)
test_loader_nn = DataLoader(test_dataset, batch_size=32, shuffle=False)


hidden_size = 50
input_len = full_dataset.features.shape[1]
model = NEOModel(input_len, hidden_size)

checkpoint_dir = Path.cwd() / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)
checkpoint_path = checkpoint_dir / "best_model.pt"

model.to(device)

learning_rate = 1e-3
epochs = 15

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    tp = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().unsqueeze(1)

            y_hat = model(X_batch)
            total_loss += loss_fn(y_hat, y_batch).item()

            probs = torch.sigmoid(y_hat)
            preds = (probs >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            tp += ((preds == 1) & (y_batch == 1)).sum().item()
            fp += ((preds == 1) & (y_batch == 0)).sum().item()
            fn += ((preds == 0) & (y_batch == 1)).sum().item()

    accuracy = correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return total_loss / len(loader), accuracy, precision, recall


best_cv_loss = float("inf")

cv_loss_list = []
cv_accuracy_list = []
training_loss_list = []

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader_nn:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        y_hat = model(X_batch)
        loss = loss_fn(y_hat, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader_nn)

    cv_loss, cv_acc, cv_prec, cv_recall = evaluate(model, cv_loader_nn, loss_fn, device)


    #matplotlib stuff
    cv_loss_list.append(cv_loss)
    cv_accuracy_list.append(cv_acc)
    training_loss_list.append(train_loss)

    saved = ""
    if cv_loss < best_cv_loss:
        best_cv_loss = cv_loss
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_len": input_len,
                "hidden_size": hidden_size,
                "epoch": epoch,
                "cv_loss": cv_loss,
            },
            checkpoint_path,
        )
        saved = "  <- saved"

    print(
        f"epoch {epoch:>2}  "
        f"train loss: {train_loss:.4f}  "
        f"cv loss: {cv_loss:.4f}  "
        f"cv acc: {cv_acc:.4f}  "
        f"cv prec: {cv_prec:.4f}  "
        f"cv recall: {cv_recall:.4f}"
        f"{saved}"
    )
    full_dataset.get_error_summary()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
x = list(range(epochs))           

ax1.plot(x, cv_loss_list, label="CV Loss", color="blue", linestyle="-", marker="o")
ax1.plot(x, training_loss_list, label="Train Loss", color="red", linestyle="-", marker="o")

ax1.set_title(f"Train and CV Loss Over {epochs} Epochs")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(x, cv_accuracy_list, label="CV Accuracy", color="blue", linestyle="-", marker="o")

ax2.set_title(f"CV Accuracy Over {epochs} Epochs")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")
ax2.legend()

plt.show()

if RUN_TEST:
    checkpoint = torch.load(checkpoint_path)
    best_model = NEOModel(checkpoint["input_len"], checkpoint["hidden_size"])
    best_model.load_state_dict(checkpoint["model_state_dict"])
    best_model.to(device)

    test_loss, test_acc, test_prec, test_recall = evaluate(
        best_model, test_loader_nn, loss_fn, device
    )
    print(
        f"\nFINAL TEST (best epoch {checkpoint['epoch']})  "
        f"loss: {test_loss:.4f}  acc: {test_acc:.4f}  "
        f"prec: {test_prec:.4f}  recall: {test_recall:.4f}"
    )
