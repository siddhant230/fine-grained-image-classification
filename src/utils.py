from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score

import torch
from torch.utils.data import Dataset, DataLoader

from dataset import CommonDataset


def get_performance(model, dataloader, device='cuda', num_classes=20, double_headed_encoder=False):
    label_list = None
    pred_list = None
    pred_probab = None

    loop = tqdm(dataloader, total=len(dataloader), leave=True)
    for (imgs, labels, label_name) in loop:
        imgs = imgs.to(device)
        labels = labels.to(device)

        if not double_headed_encoder:
            outputs = model(imgs)
        else:
            # for CLIP
            outputs, img_ftrs, text_ftrs = model(imgs, label_name)

        if labels.ndim > 1:
            labels = torch.max(labels, 1)[1]

        if label_list is None:
            label_list = labels.detach().cpu().numpy()
        else:
            label_list = np.concatenate([label_list,
                                         labels.detach().cpu().numpy()])

        softmaxed_outputs = F.softmax(outputs.detach().cpu()).numpy()
        if pred_list is None:
            pred_list = softmaxed_outputs.argmax(axis=1)
            pred_probab = softmaxed_outputs
        else:
            pred_list = np.concatenate([pred_list,
                                        softmaxed_outputs.argmax(axis=1)])
            pred_probab = np.concatenate([pred_probab,
                                          softmaxed_outputs])

    print("\n\n")
    print(classification_report(label_list, pred_list))
    # mAP

    thresholds = np.arange(start=0.2, stop=0.9, step=0.05)

    map_dict = {}
    for cls in range(num_classes):
        pred_scores = pred_probab[:, cls]

        y_true = []
        for y in label_list:
            if y == cls:
                y_true.append("positive")
            else:
                y_true.append("negative")

        precisions, recalls = precision_recall_curve(y_true=y_true,
                                                     pred_scores=pred_scores,
                                                     thresholds=thresholds,
                                                     num_classes=num_classes)
        precisions.append(1)
        recalls.append(0)
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        AP = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
        map_dict[cls] = AP

    return map_dict


def precision_recall_curve(y_true, pred_scores, thresholds, num_classes=20):
    precisions = []
    recalls = []

    for threshold in thresholds:
        y_pred = ["positive" if score >=
                  threshold else "negative" for score in pred_scores]

        precision = precision_score(
            y_true=y_true, y_pred=y_pred, pos_label="positive")
        recall = recall_score(
            y_true=y_true, y_pred=y_pred, pos_label="positive")

        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls


def get_dataloaders(names=['bottle'], verbose=1,
                    train_batch_size=128, val_batch_size=16,
                    config_dict={
                        "bottle": {"path": "./data/bottle/", "split": 0},
                        "activity": {"path": "./data/activity", "split": 0},
                        "context": {"path": "./data/context", "split": 0}},
                    ):

    output_dict = {}

    for d_type in names:

        data_path = config_dict[d_type]['path']
        train_dataset = None
        val_dataset = None

        if d_type == "bottle":
            train_dataset = CommonDataset(data_path, mode='train',
                                          split=config_dict[d_type]['split'],
                                          verbose=verbose)

            val_dataset = CommonDataset(data_path, mode='test',
                                        split=config_dict[d_type]['split'],
                                        verbose=verbose)
        elif d_type == "activity":
            pass

        else:
            train_dataset = CommonDataset(data_path, mode='train',
                                          split=config_dict[d_type]['split'],
                                          verbose=verbose)

            val_dataset = CommonDataset(data_path, mode='test',
                                        split=config_dict[d_type]['split'],
                                        verbose=verbose)

        if train_dataset is None or val_dataset is None:
            output_dict[d_type] = {"train": None,
                                   "test": None, "num_classes": None}

        else:
            train_dataloader = DataLoader(
                dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
            val_dataloader = DataLoader(
                dataset=val_dataset, batch_size=val_batch_size, shuffle=True)

            output_dict[d_type] = {"train": train_dataloader,
                                   "test": val_dataloader, "num_classes": train_dataset.num_classes}

    return output_dict
