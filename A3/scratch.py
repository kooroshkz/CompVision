import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import pandas as pd
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ======================================================
# DATASET: EVEN SAMPLING OF FRAMES
# ======================================================
class JesterDatasetMulti(Dataset):
    def __init__(self, csv_path, video_dir, labels_path, transform=None, num_frames=8):
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

        # EVEN sampling
        if total >= self.num_frames:
            step = total // self.num_frames
            selected = [frames[i * step] for i in range(self.num_frames)]
        else:
            selected = frames + [frames[-1]] * (self.num_frames - total)

        imgs = []
        for f in selected:
            img_path = os.path.join(self.video_dir, vid_id, f)
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        return torch.stack(imgs), torch.tensor(label, dtype=torch.long)


# ======================================================
# 3D CNN FROM SCRATCH (Improved architecture)
# ======================================================
class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2))
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2))
        )

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # [B, F, C, H, W] -> [B, C, F, H, W]
        x = x.permute(0, 2, 1, 3, 4)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)


# ======================================================
# VALIDATION FUNCTION
# ======================================================
def validate(model, loader, labels, device):
    model.eval()
    correct = 0
    total = 0

    num_classes = len(labels)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for frames, targets in loader:
            frames = frames.to(device)
            targets = targets.to(device)

            outputs = model(frames)
            _, preds = torch.max(outputs, 1)

            correct += (preds == targets).sum().item()
            total += targets.size(0)

            for t, p in zip(targets, preds):
                class_total[t.item()] += 1
                if t.item() == p.item():
                    class_correct[t.item()] += 1

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = 100.0 * correct / total
    print(f"\nValidation Accuracy: {accuracy:.2f}%\n")

    # Per-class accuracy
    print("Per-class accuracy:")
    for i, name in enumerate(labels):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
        else:
            acc = 0.0
        print(f"{name}: {acc:.2f}%")

    # Confusion Matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(all_targets, all_preds):
        cm[t][p] += 1

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix (3D CNN Scratch)")
    plt.savefig("scratch_confusion_matrix.png", dpi=300)
    plt.close()

    print("\nSaved confusion matrix as scratch_confusion_matrix.png\n")

    return accuracy


# ======================================================
# TRAIN + VALIDATE
# ======================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    labels = open("data/jester-v1-labels.csv").read().splitlines()

    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
    ])

    train_set = JesterDatasetMulti(
        csv_path="data/jester-v1-small-train.csv",
        video_dir="data/videos",
        labels_path="data/jester-v1-labels.csv",
        transform=transform,
        num_frames=8
    )

    val_set = JesterDatasetMulti(
        csv_path="data/jester-v1-validation.csv",
        video_dir="data/videos",
        labels_path="data/jester-v1-labels.csv",
        transform=transform,
        num_frames=8
    )

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    model = Simple3DCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 20

    print("\n=== TRAINING START ===\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for frames, labels_batch in train_loader:
            frames = frames.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} | Loss: {total_loss / len(train_loader):.4f}")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"scratch_epoch{epoch}.pth")

    torch.save(model.state_dict(), "scratch_final_model.pth")
    print("\nSaved scratch_final_model.pth\n")

    print("\n=== VALIDATING MODEL ===\n")
    validate(model, val_loader, labels, device)


if __name__ == "__main__":
    main()
