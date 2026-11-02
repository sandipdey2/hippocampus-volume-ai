"""
This module represents a UNet experiment and contains a class that handles
the experiment lifecycle
"""
import os
import time

import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_prep.SlicesDataset import SlicesDataset
from utils.utils import log_to_tensorboard
from utils.volume_stats import Dice3d, Jaccard3d
from networks.RecursiveUNet import UNet
from inference.UNetInferenceAgent import UNetInferenceAgent


class UNetExperiment:
    def __init__(self, config, split, dataset):
        self.n_epochs = config.n_epochs
        self.split = split
        self._time_start = ""
        self._time_end = ""
        self.epoch = 0
        self.name = config.name

        dirname = f'{time.strftime("%Y-%m-%d_%H%M", time.gmtime())}_{self.name}'
        self.out_dir = os.path.join(config.test_results_dir, dirname)
        os.makedirs(self.out_dir, exist_ok=True)

        self.train_loader = DataLoader(
            SlicesDataset(dataset[split["train"]]),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.val_loader = DataLoader(
            SlicesDataset(dataset[split["val"]]),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        self.test_data = dataset[split["test"]]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            print("WARNING: No CUDA device is found. This may take significantly longer!")

        self.model = UNet(num_classes=3)
        self.model.to(self.device)

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")

        self.tensorboard_train_writer = SummaryWriter(comment="_train")
        self.tensorboard_val_writer = SummaryWriter(comment="_val")

    def train(self):
        print(f"Training epoch {self.epoch}...")
        self.model.train()

        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            # TASK: put slices as 4D tensors
            data = batch["image"].to(self.device)                  # [B, 1, H, W]
            target = batch["seg"].to(self.device).squeeze(1)       # [B, H, W]

            prediction = self.model(data)
            loss = self.loss_function(prediction, target)

            loss.backward()
            self.optimizer.step()

            if i % 10 == 0:
                print(f"Batch {i}. Data shape {data.shape} Loss {loss.item():.4f}")

            # Log to TensorBoard periodically
            if i % 50 == 0:
                prediction_softmax = torch.softmax(prediction, dim=1)
                log_to_tensorboard(
                    self.tensorboard_train_writer,
                    loss.item(),
                    data,
                    target.unsqueeze(1),
                    prediction_softmax,
                    prediction,
                    self.epoch * len(self.train_loader) + i,
                )

        print(f"Training epoch {self.epoch} complete")

    def validate(self):
        print(f"Validating epoch {self.epoch}...")
        self.model.eval()
        loss_list = []

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                data = batch["image"].to(self.device)
                target = batch["seg"].to(self.device).squeeze(1)

                prediction = self.model(data)
                loss = self.loss_function(prediction, target)
                prediction_softmax = torch.softmax(prediction, dim=1)

                loss_list.append(loss.item())
                print(f"Batch {i}. Data shape {data.shape} Loss {loss.item():.4f}")

        mean_loss = np.mean(loss_list)
        self.scheduler.step(mean_loss)

        log_to_tensorboard(
            self.tensorboard_val_writer,
            mean_loss,
            data,
            target.unsqueeze(1),
            prediction_softmax,
            prediction,
            (self.epoch + 1) * 100,
        )
        print("Validation complete")

    def save_model_parameters(self):
        path = os.path.join(self.out_dir, "model.pth")
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model_parameters(self, path=""):
        if not path:
            model_path = os.path.join(self.out_dir, "model.pth")
        else:
            model_path = path
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            raise Exception(f"Could not find path {model_path}")

    def run_test(self):
        print("Testing...")
        self.model.eval()

        inference_agent = UNetInferenceAgent(model=self.model, device=self.device)

        out_dict = {"volume_stats": []}
        dc_list = []
        jc_list = []

        for i, x in enumerate(self.test_data):
            pred_label = inference_agent.single_volume_inference(x["image"])

            dc = Dice3d(pred_label, x["seg"])
            jc = Jaccard3d(pred_label, x["seg"])
            dc_list.append(dc)
            jc_list.append(jc)

            out_dict["volume_stats"].append({
                "filename": x["filename"],
                "dice": float(dc),
                "jaccard": float(jc),
            })
            print(f"{x['filename']} Dice {dc:.4f}. {100 * (i + 1) / len(self.test_data):.2f}% complete")

        out_dict["overall"] = {
            "mean_dice": float(np.mean(dc_list)),
            "mean_jaccard": float(np.mean(jc_list)),
        }
        print("\nTesting complete.")
        print(f"Mean Dice: {out_dict['overall']['mean_dice']:.4f}")
        print(f"Mean Jaccard: {out_dict['overall']['mean_jaccard']:.4f}")
        return out_dict

    def run(self):
        self._time_start = time.time()
        print("Experiment started.")

        for self.epoch in range(self.n_epochs):
            self.train()
            self.validate()

        self.save_model_parameters()

        self._time_end = time.time()
        elapsed = self._time_end - self._time_start
        print(f"Run complete. Total time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
