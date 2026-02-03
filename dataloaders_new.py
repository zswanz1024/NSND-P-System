import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import torchvision.transforms as transforms
from collections import defaultdict
import pandas as pd
# ==========================
# Custom Dataset
# ==========================
class CustomDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        csv_path = os.path.join(root, f"{split}/labels.csv")
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]  #
        #print(type(row), row)  #
        img_path = row["path"]  #
        #print(type(img_path), img_path)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = int(row["label"])
        return img, label

# ==========================
# Balanced Patient-Level Batch Sampler
# ==========================
class PatientBalancedBatchSampler(Sampler):
    def __init__(self, df, batch_size, pos_ratio=0.5):
        self.df = df
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio
        self.patient_slices = defaultdict(list)
        for idx, row in df.iterrows():
            self.patient_slices[row["patient_id"]].append(idx)

        self.indices = []
        for idx_list in self.patient_slices.values():
            self.indices.extend(idx_list)
        np.random.shuffle(self.indices)

    def __iter__(self):
        for idx in self.indices:
            yield idx  #

    def __len__(self):
        return len(self.indices)


# ==========================
# Custom Dataloader
# ==========================
class CustomDataloader_patient_sample:
    def __init__(self, batch_size, num_workers, img_resize, root_dir, pos_ratio=0.5):
        self.root = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pos_ratio = pos_ratio

        self.transform = transforms.Compose([
            transforms.Resize((img_resize, img_resize)),
            transforms.ToTensor(),
        ])
        # ===== Train Transform=====
        self.train_transform = transforms.Compose([
            transforms.Resize((img_resize, img_resize)),
            transforms.ColorJitter(
                brightness=0.1,  # ≈ tf.image.random_brightness(max_delta=0.1)
                contrast=(0.9, 1.1),  # ≈ tf.image.random_contrast
                #saturation=(0.9, 1.1),  # ≈ tf.image.random_saturation
                #hue=0.1  # ≈ tf.image.random_hue(max_delta=0.1)
            ),
            transforms.ToTensor(),  # 自动 /255
        ])

        # ===== Val / Test Transform（不做增强）=====
        self.eval_transform = transforms.Compose([
            transforms.Resize((img_resize, img_resize)),
            transforms.ToTensor(),
        ])
    def run(self, split):
        dataset = CustomDataset(
            root=self.root,
            split=split,
            transform=self.transform
        )

        if split == "train":
            sampler = PatientBalancedBatchSampler(
                df=dataset.df,
                batch_size=self.batch_size,
                pos_ratio=self.pos_ratio
            )

            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                drop_last=True  # 保证每个 batch 满
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )

        return loader, dataset
