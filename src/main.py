import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import classification_report

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
import torch.nn.functional as F

from dataset import BottleDataset
from models import CustomFrozenModel, CustomSimpleClassifier, TaskModel
from utils import get_performance


class Trainer:
    def __init__(self, model, data_path="./data/unrolled", output_dir="./outputs", epochs=10):
        self.device = torch.device(
            "cuda:0") if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        print(summary(self.model, (3, 224, 224)))
        self.data_path = data_path
        self.train_batch_size = 8
        self.val_batch_size = 4
        self.train_dataloader, self.val_dataloader = self.get_dataloaders()
        self.output_dir = output_dir
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.callback_criteria = "val_loss"
        self.global_losses = {"train_loss": 1e10, "val_loss": 1e10}
        self.best_model = None

    def get_dataloaders(self):
        # Train Loader
        train_dataset = BottleDataset(self.data_path, split='train')
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=self.train_batch_size, shuffle=True)

        # Validation Loader
        val_dataset = BottleDataset(self.data_path, split='val')
        val_dataloader = DataLoader(
            dataset=val_dataset, batch_size=self.val_batch_size, shuffle=True)
        return train_dataloader, val_dataloader

    def custom_callbacks(self, epoch, local_losses):
        save_epoch_path = os.path.join(self.output_dir, f"{epoch}")
        if not os.path.exists(save_epoch_path):
            os.makedirs(save_epoch_path)

        self.global_losses = deepcopy(local_losses)
        self.best_model = deepcopy(self.model)

        save_dict = {
            "epoch": epoch,
            "train_loss": local_losses['train_loss'],
            "val_loss": local_losses['val_loss'],
            "model": self.best_model.state_dict()
        }
        torch.save(save_dict, save_epoch_path + f"/{epoch}.pth")

    def train(self):
        # self.model.train()

        running_losses = {"train_loss": 0, "val_loss": 0}
        for epoch in range(self.epochs):
            local_losses = {"train_loss": 0, "val_loss": 0}

            # training loop
            loop = tqdm(self.train_dataloader, total=len(
                self.train_dataloader), leave=True)
            for imgs, labels in loop:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                local_losses['train_loss'] += loss.detach().cpu()
                running_losses['train_loss'] += loss.detach().cpu()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loop.set_description(f"Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix({f"train loss": loss.item()})

            # validation loop
            with torch.no_grad():
                loop = tqdm(self.val_dataloader, total=len(
                    self.val_dataloader), leave=True)
                for imgs, labels in loop:
                    imgs = imgs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, labels)
                    local_losses['val_loss'] += loss.detach().cpu()
                    running_losses['val_loss'] += loss.detach().cpu()
                    loop.set_description(f"Epoch [{epoch}/{self.epochs}]")
                    loop.set_postfix({f"val loss": loss.item()})

            if epoch % 20 == 0:
                print(
                    f"Epoch [{epoch}/{self.epochs}] : losses: {running_losses}")
                running_losses = {"train_loss": 0, "val_loss": 0}

            if local_losses[self.callback_criteria] < self.global_losses[self.callback_criteria]:
                self.custom_callbacks(epoch=epoch,
                                      local_losses=local_losses)

            print(local_losses)


if __name__ == "__main__":

    val_dataset = BottleDataset("./data/unrolled", split='val')
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=16, shuffle=True)

    custom_model = CustomFrozenModel()
    train_obj = Trainer(model=custom_model)
    train_obj.train()
    # testing performance
    get_performance(train_obj.model, val_dataloader)

    simple_classifier = CustomSimpleClassifier()
    train_obj = Trainer(model=simple_classifier)
    train_obj.train()
    # testing performance
    get_performance(train_obj.model, val_dataloader)

    task_model = TaskModel()
    train_obj = Trainer(model=task_model)
    train_obj.train()
    # testing performance
    get_performance(train_obj.model, val_dataloader)
