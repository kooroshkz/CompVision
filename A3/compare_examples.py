import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import torchvision.models.video as video_models
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


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
        folder = os.path.join(self.video_dir, vid_id)
        frames = sorted(os.listdir(folder))
        total = len(frames)

        # even sampling
        if total >= self.num_frames:
            step = total // self.num_frames
            selected = [frames[i * step] for i in range(self.num_frames)]
        else:
            selected = frames + [frames[-1]] * (self.num_frames - total)

        imgs = []
        for f in selected:
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        clip = torch.stack(imgs).permute(1, 0, 2, 3)  # [C,T,H,W]

        return clip, torch.tensor(label)


class BaselineModel(nn.Module):
    """Baseline = single-frame ResNet18."""
    def __init__(self, num_classes=27):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape

        # take the FIRST frame (the real baseline only uses 1 frame)
        x = x[:, :, 0]        # -> [B, C, H, W]

        return self.model(x)


class R3DModel(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()
        self.net = video_models.r3d_18(weights=None)
        self.net.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.net(x)


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor()
    ])

    # ===== Load validation dataset =====
    dataset = JesterDataset3D(
        csv_path="data/jester-v1-validation.csv",
        video_dir="data/videos",
        labels_path="data/jester-v1-labels.csv",
        transform=transform,
        num_frames=16
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    labels = dataset.labels

    baseline = BaselineModel().to(device)
    baseline.load_state_dict(torch.load("baseline_model.pth", map_location=device))
    improved = R3DModel().to(device)
    improved.load_state_dict(torch.load("final_model_fullset.pth", map_location=device))

    baseline.eval()
    improved.eval()

    examples = []

    with torch.no_grad():
        for i, (clip, label) in enumerate(loader):
            clip = clip.to(device)
            label = label.to(device)

            # baseline
            b_logits = baseline(clip)
            b_pred = b_logits.argmax(1).item()
            b_conf = torch.softmax(b_logits, dim=1)[0, b_pred].item()

            # improved
            i_logits = improved(clip)
            i_pred = i_logits.argmax(1).item()
            i_conf = torch.softmax(i_logits, dim=1)[0, i_pred].item()

            true_label = label.item()

            # find cases where improved fixes baseline
            if b_pred != true_label and i_pred == true_label:
                # Get the first frame for visualization from the dataset
                video_idx = i  # current index in the dataset
                vid_id = str(dataset.df.iloc[video_idx, 0])
                folder = os.path.join(dataset.video_dir, vid_id)
                frames = sorted(os.listdir(folder))
                first_frame = Image.open(os.path.join(folder, frames[0])).convert("RGB")
                
                examples.append((first_frame, true_label, b_pred, b_conf, i_pred, i_conf))

            if len(examples) == 4:
                break

    if len(examples) == 0:
        print("No examples where improved model corrects the baseline.")
        return
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (frame, true_lbl, b_pred, b_conf, i_pred, i_conf) in zip(axs.flatten(), examples):
        ax.imshow(frame)
        ax.axis("off")
        ax.set_title(
            f"True: {labels[true_lbl]}\n"
            f"Baseline: {labels[b_pred]} ({b_conf*100:.1f}%)\n"
            f"Improved: {labels[i_pred]} ({i_conf*100:.1f}%)",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig("comparison_examples.png")
    print("Saved comparison_examples.png")


if __name__ == "__main__":
    main()
