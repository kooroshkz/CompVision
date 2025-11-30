import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import pandas as pd
import os
import random
from PIL import Image
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# === same dataset class as training ===
class JesterDataset(Dataset):
    def __init__(self, csv_path, video_dir, labels_path, transform=None):
        self.df = pd.read_csv(csv_path, sep=";", header=None)
        self.video_dir = video_dir
        self.transform = transform

        self.labels = open(labels_path).read().splitlines()
        self.label_map = {name: i for i, name in enumerate(self.labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        vid_id = str(self.df.iloc[idx, 0])
        label_name = self.df.iloc[idx, 1]
        label = self.label_map[label_name]

        frames = os.listdir(os.path.join(self.video_dir, vid_id))
        frame = random.choice(frames)

        img_path = os.path.join(self.video_dir, vid_id, frame)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# === baseline model again ===
import torchvision.models as models

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    val_set = JesterDataset(
        csv_path="data/jester-v1-validation.csv",
        video_dir="data/videos",
        labels_path="data/jester-v1-labels.csv",
        transform=transform
    )

    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    model = BaselineCNN()
    model.load_state_dict(torch.load("baseline_model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # overall accuracy
    accuracy = (all_preds == all_labels).mean()
    print("Validation Accuracy:", round(accuracy * 100, 2), "%")

    # per-class accuracy
    labels_list = val_set.labels
    per_class = {}

    for i, name in enumerate(labels_list):
        mask = (all_labels == i)
        if mask.sum() == 0:
            continue
        per_class[name] = (all_preds[mask] == all_labels[mask]).mean()

    print("\nPer-class accuracy:")
    for k, v in per_class.items():
        print(k, ":", round(v * 100, 2), "%")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", square=True)
    plt.title("Confusion Matrix")
    plt.savefig("baseline_confusion_matrix.png")
    print("\nSaved baseline_confusion_matrix.png")


if __name__ == "__main__":
    main()
