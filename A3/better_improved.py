import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models.video as video_models
import pandas as pd
import os
import random
from PIL import Image


# ===============================================================
# DATASET: EVEN SAMPLING + 16 FRAMES + LIGHT AUGMENTATION
# ===============================================================
class JesterDataset3D(Dataset):
    def __init__(self, csv_path, video_dir, labels_path, transform=None, num_frames=16, augment=True):
        self.df = pd.read_csv(csv_path, sep=";", header=None)
        self.video_dir = video_dir
        self.transform = transform
        self.num_frames = num_frames
        self.augment = augment

        self.labels = open(labels_path).read().splitlines()
        self.label_map = {name: i for i, name in enumerate(self.labels)}

        # Light augmentation
        self.aug_tf = T.ColorJitter(brightness=0.2, contrast=0.2) if augment else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        vid_id = str(self.df.iloc[idx, 0])
        label = self.label_map[self.df.iloc[idx, 1]]

        frames = sorted(os.listdir(os.path.join(self.video_dir, vid_id)))
        total = len(frames)

        # Even sampling
        if total >= self.num_frames:
            step = total // self.num_frames
            selected = [frames[i * step] for i in range(self.num_frames)]
        else:
            selected = frames + [frames[-1]] * (self.num_frames - total)

        imgs = []
        for f in selected:
            img = Image.open(os.path.join(self.video_dir, vid_id, f)).convert("RGB")
            if self.aug_tf:
                img = self.aug_tf(img)
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        # shape: [C, T, H, W]
        frames_tensor = torch.stack(imgs).permute(1, 0, 2, 3)

        return frames_tensor, torch.tensor(label)


# ===============================================================
# MODEL: R3D-18 (3D RESNET)
# ===============================================================
class R3DModel(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()
        self.net = video_models.r3d_18(weights=None)
        self.net.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.net(x)


# ===============================================================
# TRAINING LOOP
# ===============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    transform = T.Compose([
        T.Resize((112, 112)),   # recommended for 3D ResNet
        T.ToTensor()
    ])

    # FULL BIG TRAIN SET
    train_set = JesterDataset3D(
        csv_path="data/jester-v1-train.csv",
        video_dir="data/videos",
        labels_path="data/jester-v1-labels.csv",
        transform=transform,
        num_frames=16,
        augment=True
    )

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)

    model = R3DModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 20

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for vid, labels in train_loader:
            vid = vid.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(vid)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss / len(train_loader)}")

        # Save safe checkpoints
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"final_model_epoch{epoch}.pth")

    torch.save(model.state_dict(), "final_model_fullset.pth")
    print("Saved final_model_fullset.pth")


if __name__ == "__main__":
    main()
