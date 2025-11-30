import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import pandas as pd
import numpy as np
import os, random
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
#     Helpers
# ============================================================

def safe_print(*args):
    try:
        print(*args)
    except:
        pass


def save_conf_mat(cm, name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues")
    plt.title(name)
    plt.savefig(name.replace(" ", "_") + ".png")
    plt.close()


def validate_model(model, val_loader, labels_list, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            out = model(imgs)
            preds = out.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()

    # per-class accuracy
    per_class = {}
    for i, name in enumerate(labels_list):
        mask = (all_labels == i)
        if mask.sum() > 0:
            per_class[name] = (all_preds[mask] == all_labels[mask]).mean()

    cm = confusion_matrix(all_labels, all_preds)

    return acc, per_class, cm


# ============================================================
#     Dataset variations
# ============================================================

class MultiFrameDataset(Dataset):
    """Base dataset supporting different sampling strategies."""
    def __init__(self, csv, video_dir, labels_path, transform, num_frames=8, mode="even", augment=False):
        self.df = pd.read_csv(csv, sep=";", header=None)
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.transform = transform
        self.mode = mode
        self.augment = augment

        self.labels = open(labels_path).read().splitlines()
        self.label_map = {n: i for i, n in enumerate(self.labels)}

        # augmentation pipeline
        if augment:
            self.aug = T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.RandomHorizontalFlip()
            ])
        else:
            self.aug = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        vid = str(self.df.iloc[idx, 0])
        label = self.label_map[self.df.iloc[idx, 1]]

        path = os.path.join(self.video_dir, vid)
        frames = sorted(os.listdir(path))
        total = len(frames)

        # ----- Option: even sampling -----
        if self.mode == "even" and total >= self.num_frames:
            step = total // self.num_frames
            selected = [frames[i * step] for i in range(self.num_frames)]

        # ----- Option: random sampling -----
        else:
            if total >= self.num_frames:
                selected = sorted(random.sample(frames, self.num_frames))
            else:
                selected = frames + [frames[-1]] * (self.num_frames - total)

        imgs = []
        for f in selected:
            img = Image.open(os.path.join(path, f)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            if self.aug:
                img = self.aug(img)
            imgs.append(img)

        return torch.stack(imgs), torch.tensor(label)


# ============================================================
#     Models
# ============================================================

# ---- Temporal averaging model ----
class MultiFrameResNet(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()
        self.net = models.resnet18(weights="IMAGENET1K_V1")
        self.net.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        B, F, C, H, W = x.shape
        x = x.view(B * F, C, H, W)
        out = self.net(x)
        out = out.view(B, F, -1).mean(dim=1)
        return out


# ---- 3D CNN ----
class R3DNet(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()
        self.net = models.video.r3d_18(weights=None)
        self.net.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # expects (B, C, F, H, W)
        x = x.transpose(1, 2)
        return self.net(x)


# ============================================================
#     Training
# ============================================================

def train_model(model, train_loader, device, epochs=3):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        total = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            opt.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, labels)
            loss.backward()
            opt.step()

            total += loss.item()

        safe_print("Epoch", ep, "Loss:", total / len(train_loader))

    return model


# ============================================================
#     Full Experiment Runner
# ============================================================

def run_experiment(name, dataset_args):
    safe_print("\n============================")
    safe_print(f"Running: {name}")
    safe_print("============================")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

        train_set = MultiFrameDataset(
            csv="data/jester-v1-small-train.csv",
            video_dir="data/videos",
            labels_path="data/jester-v1-labels.csv",
            transform=transform,
            **dataset_args
        )
        val_set = MultiFrameDataset(
            csv="data/jester-v1-validation.csv",
            video_dir="data/videos",
            labels_path="data/jester-v1-labels.csv",
            transform=transform,
            **dataset_args
        )

        train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

        # pick model
        if name == "D) 3D ResNet-18":
            model = R3DNet()
        else:
            model = MultiFrameResNet()

        model = train_model(model, train_loader, device, epochs=3)

        acc, per_class, cm = validate_model(model, val_loader, val_set.labels, device)

        safe_print(f"\n{name} Accuracy: {round(acc * 100, 2)}%")
        save_conf_mat(cm, f"{name} Confusion Matrix")

        return acc

    except Exception as e:
        safe_print(f"FAILED: {name} because:", str(e))
        return 0.0


# ============================================================
#     MAIN
# ============================================================

def main():
    results = {}

    # A) even 8 frames
    results["A"] = run_experiment("A) Even 8-frame", {
        "num_frames": 8,
        "mode": "even",
        "augment": False
    })

    # B) random 16 frames
    results["B"] = run_experiment("B) Random 16-frame", {
        "num_frames": 16,
        "mode": "random",
        "augment": False
    })

    # C) 8 frames + augmentation
    results["C"] = run_experiment("C) 8-frame + Augment", {
        "num_frames": 8,
        "mode": "random",
        "augment": True
    })

    # D) 3D ResNet
    results["D"] = run_experiment("D) 3D ResNet-18", {
        "num_frames": 16,
        "mode": "even",
        "augment": False
    })

    # Summary
    safe_print("\n\n===== FINAL SUMMARY =====")
    for k, v in results.items():
        safe_print(f"{k}: {round(v * 100, 2)}%")


if __name__ == "__main__":
    main()