"""
Module for U-Net inference agent
"""
import numpy as np
import torch


class UNetInferenceAgent:
    def __init__(self, model, device=None):
        self.model = model
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model.to(self.device)

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
