import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import pandas as pd
import os
import random
from PIL import Image


# ===== Improved Dataset: sample 8 frames =====
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
        label_name = self.df.iloc[idx, 1]
        label = self.label_map[label_name]

        frames = sorted(os.listdir(os.path.join(self.video_dir, vid_id)))
        total = len(frames)

        # random sampling 8 frames
        if total >= self.num_frames:
            selected = sorted(random.sample(frames, self.num_frames))
        else:
            selected = frames + [frames[-1]] * (self.num_frames - total)

        imgs = []
        for f in selected:
            img_path = os.path.join(self.video_dir, vid_id, f)
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        # return tensor [8, 3, H, W]
        imgs = torch.stack(imgs)
        return imgs, torch.tensor(label)


# ===== Improved Model: average frame logits =====
class MultiFrameResNet(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x shape = [B, F, 3, H, W]
        batch, frames, c, h, w = x.shape
        x = x.view(batch * frames, c, h, w)

        logits = self.backbone(x)  # [B*F, classes]
        logits = logits.view(batch, frames, -1)
        logits = logits.mean(dim=1)  # average over frames

        return logits


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    train_set = JesterDatasetMulti(
        csv_path="data/jester-v1-small-train.csv",
        video_dir="data/videos",
        labels_path="data/jester-v1-labels.csv",
        transform=transform,
        num_frames=8
    )

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)

    model = MultiFrameResNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(5):  # more epochs than baseline
        model.train()
        total_loss = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("Epoch:", epoch, "Loss:", total_loss / len(train_loader))

    torch.save(model.state_dict(), "improved_model.pth")
    print("Saved improved_model.pth")


if __name__ == "__main__":
    main()
