"""
Module for Pytorch dataset representations
"""
import torch
from torch.utils.data import Dataset
import numpy as np


class SlicesDataset(Dataset):
    def __init__(self, data):
        self.slices = []
        data_list = list(data)

        for d in data_list:
            img = np.asarray(d["image"])
            seg = np.asarray(d["seg"])

            # Use the smaller of the two depths
            n = min(img.shape[0], seg.shape[0])

            for j in range(n):
                self.slices.append((img[j].copy(), seg[j].copy()))

        print(f"SlicesDataset created with {len(self.slices)} slices")

    def __getitem__(self, idx):
        img_slice, seg_slice = self.slices[idx]
        sample = {"id": idx}
        sample["image"] = torch.from_numpy(img_slice[None, ...]).float()
        sample["seg"]   = torch.from_numpy(seg_slice[None, ...]).long()
        return sample

    def __len__(self):
        return len(self.slices)