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

        return img, torch.tensor(label)


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1")
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

    train_set = JesterDataset(
        csv_path="data/jester-v1-small-train.csv",
        video_dir="data/videos",
        labels_path="data/jester-v1-labels.csv",
        transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    model = BaselineCNN()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(3):
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

        print("Epoch", epoch, "Loss:", total_loss / len(train_loader))

    torch.save(model.state_dict(), "baseline_model.pth")
    print("Saved baseline_model.pth")


if __name__ == "__main__":
    main()

