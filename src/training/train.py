import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from src.data.dataset import NEODataset, get_normalization_transform
from src.models.model import NEOModel

# Flip to True only when done experimenting and want the final,
# one-time test-set number. Keep it False while tuning so you don't peek.
RUN_TEST = False

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

# Normalization stats come from the TRAINING rows only, then the same
# transform is applied to every split (all three share full_dataset).
# features is now a tensor, so index by indices and reduce along dim=0.
# NaN-aware: some columns have genuine missing values, so plain mean/std
# would be NaN. nanmean ignores them; std is computed the same way.
train_features = full_dataset.features[train_dataset.indices]
mean_vector = torch.nanmean(train_features, dim=0)
diff = train_features - mean_vector
std_vector = torch.sqrt(torch.nanmean(diff * diff, dim=0))
full_dataset.transform = get_normalization_transform(mean_vector, std_vector)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
cv_loader = DataLoader(cv_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---------------------------------------------------------------- model
hidden_size = 50
input_len = full_dataset.features.shape[1]
model = NEOModel(input_len, hidden_size)

# Where the best checkpoint goes. Kept out of git (see .gitignore).
checkpoint_dir = Path.cwd() / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)
checkpoint_path = checkpoint_dir / "best_model.pt"

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
model.to(device)

learning_rate = 1e-3
epochs = 100

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def evaluate(model, loader, loss_fn, device):
    model.eval()  
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().unsqueeze(1)

            y_hat = model(X_batch)
            total_loss += loss_fn(y_hat, y_batch).item()

            preds = (y_hat >= 0.5).float() 
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    return total_loss / len(loader), correct / total


best_cv_loss = float("inf")  

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        y_hat = model(X_batch)
        loss = loss_fn(y_hat, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    cv_loss, cv_acc = evaluate(model, cv_loader, loss_fn, device)

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
        f"cv acc: {cv_acc:.4f}"
        f"{saved}"
    )


if RUN_TEST:
    # Reload the best checkpoint so we test the best epoch, not the last one.
    checkpoint = torch.load(checkpoint_path)
    best_model = NEOModel(checkpoint["input_len"], checkpoint["hidden_size"])
    best_model.load_state_dict(checkpoint["model_state_dict"])
    best_model.to(device)

    test_loss, test_acc = evaluate(best_model, test_loader, loss_fn, device)
    print(
        f"\nFINAL TEST (best epoch {checkpoint['epoch']})  "
        f"loss: {test_loss:.4f}  acc: {test_acc:.4f}"
    )
