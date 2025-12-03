import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics import confusion_matrix


# ============================================================
# DATASET (reused)
# ============================================================
class JesterDataset3D(torch.utils.data.Dataset):
    def __init__(self, csv_path, video_dir, labels_path, transform=None, num_frames=16):
        self.df = pd.read_csv(csv_path, sep=";", header=None)
        self.video_dir = video_dir
        self.transform = transform
        self.num_frames = num_frames

        self.labels = open(labels_path).read().splitlines()
        self.label_map = {name: i for i, name in enumerate(self.labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        vid_id = str(self.df.iloc[idx, 0])
        label = self.label_map[self.df.iloc[idx, 1]]

        frames = sorted(os.listdir(os.path.join(self.video_dir, vid_id)))
        total = len(frames)

        # even sample 16 frames
        if total >= self.num_frames:
            step = total // self.num_frames
            selected = [frames[i * step] for i in range(self.num_frames)]
        else:
            selected = frames + [frames[-1]] * (self.num_frames - total)

        imgs = []
        for f in selected:
            img = Image.open(os.path.join(self.video_dir, vid_id, f)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        video_tensor = torch.stack(imgs).permute(1, 0, 2, 3)  # C,T,H,W
        return video_tensor, torch.tensor(label)


# ============================================================
# MODELS
# ============================================================
import torchvision.models.video as video_models

class R3DModel(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()
        self.net = video_models.r3d_18(weights=None)
        self.net.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.net(x)


class Baseline2D(nn.Module):
    """Baseline model: ResNet18 avg over frames"""
    def __init__(self, num_classes=27):
        super().__init__()
        import torchvision.models as models
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: [B,C,T,H,W] from JesterDataset3D
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # [B,T,C,H,W]
        x = x.reshape(B * T, C, H, W)
        logits = self.model(x)
        logits = logits.reshape(B, T, -1).mean(dim=1)
        return logits


# ============================================================
# EVAL FUNCTION
# ============================================================
def run_eval(model, loader, device):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for vid, labels in loader:
            vid = vid.to(device)
            labels = labels.to(device)

            out = model(vid)
            pred = out.argmax(1)

            preds.extend(pred.cpu().numpy())
            trues.extend(labels.cpu().numpy())

    return np.array(preds), np.array(trues)


# ============================================================
# MAIN
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    transform = T.Compose([
        T.Resize((224, 224)),  # Match the baseline training size
        T.ToTensor()
    ])

    val_set = JesterDataset3D(
        csv_path="data/jester-v1-validation.csv",
        video_dir="data/videos",
        labels_path="data/jester-v1-labels.csv",
        transform=transform,
        num_frames=16
    )

    loader = DataLoader(val_set, batch_size=2, shuffle=False)
    labels = val_set.labels

    # ===== Load models =====
    baseline = Baseline2D().to(device)
    improved = R3DModel().to(device)

    baseline.load_state_dict(torch.load("baseline_model.pth", map_location=device))
    improved.load_state_dict(torch.load("final_model_fullset.pth", map_location=device))

    # ===== Evaluate =====
    print("Running baseline...")
    b_preds, b_trues = run_eval(baseline, loader, device)

    print("Running improved model...")
    i_preds, i_trues = run_eval(improved, loader, device)

    # ============================================================
    # Confusion Matrices (side-by-side)
    # ============================================================
    plt.figure(figsize=(18, 7))

    for idx, (preds, trues, title) in enumerate([
        (b_preds, b_trues, "Baseline Model"),
        (i_preds, i_trues, "Improved R3D Model")
    ]):
        cm = confusion_matrix(trues, preds)
        plt.subplot(1, 2, idx + 1)
        sns.heatmap(cm, cmap="Blues")
        plt.title(title)

    plt.savefig("compare_confusion_matrices.png")
    print("Saved compare_confusion_matrices.png")

    # ============================================================
    # Per-class accuracy comparison
    # ============================================================
    b_acc = []
    i_acc = []

    for i in range(len(labels)):
        mask = (b_trues == i)
        b_acc.append((b_preds[mask] == b_trues[mask]).mean() * 100 if mask.sum() else 0)

        mask = (i_trues == i)
        i_acc.append((i_preds[mask] == i_trues[mask]).mean() * 100 if mask.sum() else 0)

    # plot
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(18, 6))
    plt.bar(x - width/2, b_acc, width, label="Baseline")
    plt.bar(x + width/2, i_acc, width, label="Improved R3D")
    plt.xticks(x, labels, rotation=90)
    plt.title("Per-Class Accuracy Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("compare_per_class_accuracy.png")
    print("Saved compare_per_class_accuracy.png")

    # ============================================================
    # Biggest Improvements
    # ============================================================
    diff = np.array(i_acc) - np.array(b_acc)
    top5 = diff.argsort()[-5:][::-1]
    worst5 = diff.argsort()[:5]

    # plot best improvements
    plt.figure(figsize=(7,5))
    plt.barh([labels[i] for i in top5], diff[top5], color="green")
    plt.title("Top-5 Most Improved Classes")
    plt.xlabel("Accuracy Gain (%)")
    plt.tight_layout()
    plt.savefig("compare_top5_improved.png")
    print("Saved compare_top5_improved.png")

    # plot biggest drops (if any)
    plt.figure(figsize=(7,5))
    plt.barh([labels[i] for i in worst5], diff[worst5], color="red")
    plt.title("Top-5 Accuracy Decreases")
    plt.xlabel("Accuracy Loss (%)")
    plt.tight_layout()
    plt.savefig("compare_top5_worse.png")
    print("Saved compare_top5_worse.png")

    print("\n=== ALL COMPARISON VISUALS SAVED ===")


if __name__ == "__main__":
    main()
