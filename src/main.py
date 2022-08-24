import os
import shutil
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
from models import CustomModel, ClipModel
from utils import get_performance, get_dataloaders


class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader,
                 output_dir="./outputs",
                 epochs=10, start=0, perf_k=2,
                 verbose=1, autoclean=True, d_type="bottle", num_classes=20):

        # verbose 0->no model summary, no performance
        # verbose 1->no model summary, yes performance
        # verbose 2->both

        # torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        self.model = model.to(self.device)
        self.start = start
        self.verbose = verbose
        self.autoclean = autoclean
        self.d_type = d_type
        self.num_classes = num_classes

        if self.verbose > 1:
            print(summary(self.model, (3, 224, 224)))

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.output_dir = output_dir
        self.epochs = epochs
        # optim.SGD(self.model.parameters(), lr=1e-1, momentum=0.9) #
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-4)

        weights = self.get_weight_criterion(self.d_type)
        class_weights = torch.FloatTensor(weights).cuda()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.callback_criteria = "val_loss"
        self.global_losses = {"train_loss": 1e10, "val_loss": 1e10}
        self.best_model = None
        self.perf_k = perf_k
        self.save_store = sorted(os.listdir(
            self.output_dir), key=lambda x: int(x))
        print(f"Pre-saved weights @ {self.output_dir} are {self.save_store}")

    def get_weight_criterion(self, dataset='bottles'):
        if dataset == 'context':
            weights = [1, 1, 2.2, 1, 1, 1, 1, 1, 4.5, 1, 1, 1, 1,
                       1, 1, 2, 1, 1, 1, 1, 1, 1, 2.2, 1.5, 1, 1, 1, 1]
        elif dataset == 'bottles':
            weights = [1, 1, 1, 1, 2, 1.5, 1.5, 1, 1.5,
                       1, 1, 1.5, 1, 1, 1, 1.5, 1, 1, 1, 1]
        return weights

    def custom_callbacks(self, epoch, local_losses):
        self.save_store.append(epoch)
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
        if self.autoclean:
            safe = 0
            while len(self.save_store) > 3:
                dir_to_delete = str(self.save_store.pop(0))
                if os.path.exists(os.path.join(self.output_dir, dir_to_delete)):
                    shutil.rmtree(os.path.join(self.output_dir, dir_to_delete))
                # for preventing infinite deadlock
                if safe > 10:
                    break

    def train(self, do_agg=False):

        running_losses = {"train_loss": 0, "val_loss": 0}
        for epoch in range(self.start, self.epochs):
            local_losses = {"train_loss": 0, "val_loss": 0}

            # training loop
            agg_idx = 0
            agg_loss = 0.0

            loop = tqdm(self.train_dataloader, total=len(
                self.train_dataloader), leave=True)
            for (imgs, labels, label_name) in loop:
                # print(imgs.shape, labels.shape)
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(imgs)
                if len(labels[0].shape) > 1:
                    loss = self.criterion(outputs, torch.max(labels, 1)[1])
                else:
                    loss = self.criterion(outputs, labels)

                agg_loss += loss
                agg_idx += 1
                local_losses['train_loss'] += loss.detach().cpu()
                running_losses['train_loss'] += loss.detach().cpu()

                if do_agg:
                    if agg_idx > 4:
                        self.optimizer.zero_grad()
                        agg_loss.backward()
                        agg_idx = 0
                        agg_loss = 0.0
                        self.optimizer.step()
                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                loop.set_description(
                    f"{self.d_type}@Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix({f"train loss": loss.item()})

            # validation loop
            with torch.no_grad():
                loop = tqdm(self.val_dataloader, total=len(
                    self.val_dataloader), leave=True)
                for (imgs, labels, label_name) in loop:
                    imgs = imgs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, labels)
                    local_losses['val_loss'] += loss.detach().cpu()
                    running_losses['val_loss'] += loss.detach().cpu()
                    loop.set_description(f"Epoch [{epoch}/{self.epochs}]")
                    loop.set_postfix({f"val loss": loss.item()})

            if epoch % self.perf_k == 0 and self.verbose:
                # testing performance
                get_performance(self.model, self.val_dataloader,
                                num_classes=self.num_classes)

            if epoch % 20 == 0:
                print(
                    f"Epoch [{epoch}/{self.epochs}] : losses: {running_losses}")
                running_losses = {"train_loss": 0, "val_loss": 0}

            if local_losses[self.callback_criteria] != self.global_losses[self.callback_criteria]:
                print(f"------ Saving at epoch: {epoch} ------")
                self.custom_callbacks(epoch=epoch,
                                      local_losses=local_losses)

            print(local_losses)


if __name__ == "__main__":

    dataloaders = get_dataloaders(
        ['context'], train_batch_size=128, val_batch_size=32)
    opt_dir = {"bottle": "./outputs/bottle/", "context": "./outputs/context"}
    loaded_model_weights = {"bottle": "", "context": ""}
    final_models = {}
    device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'

    for d_type in dataloaders:
        start = 0
        print("Num classes:", dataloaders[d_type]['num_classes'])
        model = CustomModel(num_classes=dataloaders[d_type]['num_classes'])
        # print(summary(model.to(device), (3,224,224)))

        if loaded_model_weights[d_type] != "":
            start = int(loaded_model_weights[d_type].split("/")[-2])
            print(f"[INFO] Loading model for {d_type} from epoch: {start}")
            model.load_state_dict(torch.load(
                loaded_model_weights[d_type])['model'])

        train_obj = Trainer(
            model=model,
            train_dataloader=dataloaders[d_type]['train'],
            val_dataloader=dataloaders[d_type]['test'],
            output_dir=opt_dir[d_type],
            start=start+1,
            epochs=10,
            d_type=d_type,
            num_classes=dataloaders[d_type]['num_classes']
        )
        train_obj.train()
        final_models[d_type] = train_obj
