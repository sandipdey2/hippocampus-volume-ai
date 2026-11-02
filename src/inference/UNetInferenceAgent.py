"""
Module for U-Net inference agent
"""
import numpy as np
import torch


class UNetInferenceAgent:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    def single_volume_inference(self, volume):
        """
        volume: numpy array of shape [X, Y, Z]
        Returns: 3D numpy array with predicted class labels
        """
        self.model.eval()
        slices = []

        with torch.no_grad():
            for i in range(volume.shape[0]):  # iterate over X (0th) dimension
                slc = volume[i, :, :]
                t = torch.from_numpy(slc[None, None, ...]).float().to(self.device)
                pred = self.model(t)
                mask = torch.argmax(pred, dim=1).cpu().numpy()[0]
                slices.append(mask)

        return np.stack(slices, axis=0)
