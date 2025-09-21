# Multi-dataset MLP experiment: MNIST / FashionMNIST / Tiny ImageNet
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- Setup ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device} ...")
torch.manual_seed(42)

BATCH_SIZE = 512
EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN = 256
PRINT_EVERY = 100

class TinyImageNetVal(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []

        ann_file = os.path.join(root, "val_annotations.txt")
        img_dir = os.path.join(root, "images")

        with open(ann_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                img_name, label = parts[0], parts[1]
                img_path = os.path.join(img_dir, img_name)
                self.samples.append((img_path, label))

        labels = sorted(list(set([s[1] for s in self.samples])))
        self.label_to_idx = {label: idx for idx, label in enumerate(labels)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = self.label_to_idx[label]
        return img, target

# ---------------- Helper ----------------
def get_loaders(dataset_name):
    """Return train/test DataLoader and input dim, num_classes."""
    if dataset_name == "MNIST":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
        test_ds  = datasets.MNIST("./data", train=False, download=True, transform=tfm)
        input_dim, num_classes = 28*28, 10

    elif dataset_name == "FashionMNIST":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_ds = datasets.FashionMNIST("./data", train=True, download=True, transform=tfm)
        test_ds  = datasets.FashionMNIST("./data", train=False, download=True, transform=tfm)
        input_dim, num_classes = 28*28, 10

    elif dataset_name == "TinyImageNet":
        tfm = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        train_ds = datasets.ImageFolder("./data/tiny-imagenet-200/train", transform=tfm)
        test_ds  = TinyImageNetVal("./data/tiny-imagenet-200/val", transform=tfm)
        input_dim, num_classes = 64*64*3, 200

    else:
        raise ValueError("Unknown dataset")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    return train_loader, test_loader, input_dim, num_classes


def build_model(input_dim, num_classes):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, HIDDEN),
        nn.ReLU(inplace=True),
        nn.Linear(HIDDEN, num_classes),
    ).to(device)

    # He init
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model

def build_model_tiny(input_dim, num_classes):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes),
    ).to(device)
    return model



def train_and_eval(dataset_name):
    train_loader, test_loader, input_dim, num_classes = get_loaders(dataset_name)
    if dataset_name == "TinyImageNet":
        model = build_model_tiny(input_dim, num_classes)
    else:
        model = build_model(input_dim, num_classes)
    lossfn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_losses, test_accs = [], []

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        running_loss, seen = 0.0, 0
        for it, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = lossfn(logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * y.size(0)
            seen += y.size(0)

        avg_loss = running_loss / seen
        train_losses.append(avg_loss)

        # ---- Eval ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = 100.0 * correct / total
        test_accs.append(acc)

        # print(f"[{dataset_name}] Epoch {epoch}/{EPOCHS} | loss={avg_loss:.4f} | acc={acc:.2f}%")

    return train_losses, test_accs


# ---------------- Run for all datasets ----------------
results = {}
for name in ["MNIST", "FashionMNIST"]:
    train_losses, test_accs = train_and_eval(name)
    results[name] = {"loss": train_losses, "acc": test_accs}
    print(f"[{name}] Final Test Accuracy: {test_accs[-1]:.2f}%")
# for name in ["MNIST", "FashionMNIST", "TinyImageNet"]:
#     train_losses, test_accs = train_and_eval(name)
#     results[name] = {"loss": train_losses, "acc": test_accs}

# ---------------- Plot ----------------
epochs = np.arange(1, EPOCHS+1)

# Loss plot
plt.figure(figsize=(8,6))
for name, res in results.items():
    plt.plot(epochs, res["loss"], marker="o", label=f"{name} Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("loss_comparison.png")
plt.close()

# Accuracy plot
plt.figure(figsize=(8,6))
for name, res in results.items():
    plt.plot(epochs, res["acc"], marker="s", label=f"{name} Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("acc_comparison.png")
plt.close()

print("Saved loss_comparison.png and acc_comparison.png")
