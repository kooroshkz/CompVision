import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import pandas as pd
import os
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import torchvision.models.video as video_models


class JesterDataset3D(Dataset):
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

        # even sampling
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

        vid_tensor = torch.stack(imgs).permute(1, 0, 2, 3)   # [C,T,H,W]
        return vid_tensor, torch.tensor(label)

class R3DModel(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()
        self.net = video_models.r3d_18(weights=None)
        self.net.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.net(x)

def main():
    print("=== Generating visuals ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor()
    ])

    val_set = JesterDataset3D(
        csv_path="data/jester-v1-validation.csv",
        video_dir="data/videos",
        labels_path="data/jester-v1-labels.csv",
        transform=transform,
        num_frames=16
    )
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False)

    model = R3DModel()
    model.load_state_dict(torch.load("final_model_fullset.pth", map_location=device))
    model = model.to(device)
    model.eval()

    print("Model loaded.")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for vid, labels in val_loader:
            vid = vid.to(device)
            labels = labels.to(device)

            out = model(vid)
            pred = out.argmax(1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    labels_list = val_set.labels


    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, cmap="Blues")
    plt.title("Confusion Matrix - R3D Model")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("vis_confusion_matrix.png")
    print("Saved vis_confusion_matrix.png")

    per_class_acc = []
    for i, name in enumerate(labels_list):
        mask = all_labels == i
        acc = (all_preds[mask] == all_labels[mask]).mean() * 100 if mask.sum() > 0 else 0
        per_class_acc.append(acc)

    plt.figure(figsize=(16, 6))
    plt.bar(range(len(labels_list)), per_class_acc)
    plt.xticks(range(len(labels_list)), labels_list, rotation=90)
    plt.ylabel("Accuracy (%)")
    plt.title("Per-Class Accuracy")
    plt.tight_layout()
    plt.savefig("vis_per_class_accuracy.png")
    print("Saved vis_per_class_accuracy.png")


    easiest_idx = np.argsort(per_class_acc)[-5:]
    easiest_names = [labels_list[i] for i in easiest_idx]
    easiest_vals = [per_class_acc[i] for i in easiest_idx]

    plt.figure(figsize=(8, 5))
    plt.barh(easiest_names, easiest_vals, color="green")
    plt.xlabel("Accuracy (%)")
    plt.title("Top-5 Easiest Classes")
    plt.tight_layout()
    plt.savefig("vis_top5_easiest.png")
    print("Saved vis_top5_easiest.png")

    hardest_idx = np.argsort(per_class_acc)[:5]
    hardest_names = [labels_list[i] for i in hardest_idx]
    hardest_vals = [per_class_acc[i] for i in hardest_idx]

    plt.figure(figsize=(8, 5))
    plt.barh(hardest_names, hardest_vals, color="red")
    plt.xlabel("Accuracy (%)")
    plt.title("Top-5 Hardest Classes")
    plt.tight_layout()
    plt.savefig("vis_top5_hardest.png")
    print("Saved vis_top5_hardest.png")

    print("\n=== All visuals saved successfully ===")


if __name__ == "__main__":
    main()
