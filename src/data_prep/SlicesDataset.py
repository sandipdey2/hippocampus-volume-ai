"""
Module for Pytorch dataset representations
"""
import torch
from torch.utils.data import Dataset


class SlicesDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.slices = []
        for i, d in enumerate(data):
            for j in range(d["image"].shape[0]):
                self.slices.append((i, j))

    def __getitem__(self, idx):
        slc = self.slices[idx]
        sample = dict()
        sample["id"] = idx

        vol_idx, slice_idx = slc
        # [1, H, W]
        sample["image"] = torch.from_numpy(
            self.data[vol_idx]["image"][slice_idx, :, :][None, ...]
        ).float()
        sample["seg"] = torch.from_numpy(
            self.data[vol_idx]["seg"][slice_idx, :, :][None, ...]
        ).long()

        return sample

    def __len__(self):
        return len(self.slices)
