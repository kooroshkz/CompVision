import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import pandas as pd
import os
from PIL import Image


# ===== Improved Dataset: EVEN sampling 12 frames =====
class JesterDatasetMulti(Dataset):
    def __init__(self, csv_path, video_dir, labels_path, transform=None, num_frames=12):
        self.df = pd.read_csv(csv_path, sep=";", header=None)
        self.video_dir = video_dir
        self.transform = transform
        self.num_frames = num_frames

        self.labels = open(labels_path).read().splitlines()
        self.label_map = {name: i for i, name in enumerate(self.labels)}

        # light augmentation (safe)
        self.augment = T.ColorJitter(brightness=0.2, contrast=0.2)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        vid_id = str(self.df.iloc[idx, 0])
        label_name = self.df.iloc[idx, 1]
        label = self.label_map[label_name]

        frames = sorted(os.listdir(os.path.join(self.video_dir, vid_id)))
        total = len(frames)

        # ===== EVEN SAMPLING =====
        if total >= self.num_frames:
            step = total // self.num_frames
            selected = [frames[i * step] for i in range(self.num_frames)]
        else:
            selected = frames + [frames[-1]] * (self.num_frames - total)

        imgs = []
        for f in selected:
            img_path = os.path.join(self.video_dir, vid_id, f)
            img = Image.open(img_path).convert("RGB")

            # resize + crop = BEST practice
            if self.transform:
                img = self.transform(img)

            # safe jitter: brightness + contrast only
            img = self.augment(img)

            imgs.append(img)

        # [F, 3, H, W]
        imgs = torch.stack(imgs)
        return imgs, torch.tensor(label)


# ===== Improved Model: ResNet34 + temporal averaging =====
class MultiFrameResNet34(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()
        self.backbone = models.resnet34(weights="IMAGENET1K_V1")
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        B, F, C, H, W = x.shape
        x = x.view(B * F, C, H, W)
        logits = self.backbone(x)
        logits = logits.view(B, F, -1)
        return logits.mean(dim=1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor()
    ])

    train_set = JesterDatasetMulti(
        csv_path="data/jester-v1-small-train.csv",
        video_dir="data/videos",
        labels_path="data/jester-v1-labels.csv",
        transform=transform,
        num_frames=12          # best option
    )

    train_loader = DataLoader(train_set, batch_size=6, shuffle=True)  # ResNet34 uses more memory

    model = MultiFrameResNet34().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(6):   # small extra training helps ResNet34
        model.train()
        total_loss = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("Epoch:", epoch, "Loss:", total_loss / len(train_loader))

    torch.save(model.state_dict(), "improved_model_final.pth")
    print("Saved improved_model_final.pth")


if __name__ == "__main__":
    main()