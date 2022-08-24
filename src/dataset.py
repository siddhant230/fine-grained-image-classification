import os
import cv2
import json
import random
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CommonDataset(Dataset):
    def __init__(self, data_path="./data", split=0, mode="train", verbose=1):

        self.split = split
        self.mode = mode
        self.data_path = data_path

        print(f"[INFO]: split: {split} | path: {data_path}")
        print(
            f"[INFO]: mode: {mode} | split_file: {self.data_path +'/split_'+ str(self.split) +'.json'}")

        with open(self.data_path + '/split_' + str(self.split) + '.json', 'r') as fp:
            self.gt_annotations = json.load(fp)

        with open(self.data_path + '/classes.json', 'r') as fp:
            self.classes_labels = json.load(fp)

        if self.mode == "train":
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        self.image_list = list(self.gt_annotations[self.mode].keys())
        self.num_classes = len(
            Counter(self.gt_annotations[self.mode].values()))
        print(f"Total num classes: {self.num_classes}")
        if verbose:
            self.visalize()

    def visalize(self, w=50, h=50, columns=4, rows=5):
        w = 20
        h = 20
        fig = plt.figure(figsize=(8, 8))
        for i in range(1, columns*rows + 1):
            idx = np.random.randint(1, len(self.image_list)-1)
            img = cv2.imread(os.path.join(
                self.data_path, self.image_list[idx]))
            fig.add_subplot(rows, columns, i)
            plt.title(
                self.classes_labels[self.gt_annotations[self.mode][self.image_list[idx]]])
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.imshow(img)

        plt.show()
        data_dist = Counter(self.gt_annotations[self.mode].values())
        print("\nclass distribution with data size of: ", sum(data_dist.values()))
        plt.figure(figsize=(4, 1))
        plt.bar(range(len(data_dist)), list(
            data_dist.values()), align='center')
        plt.show()
        print("----------------------------------------------------")

    def __len__(self):
        return len(self.gt_annotations[self.mode])

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        img = Image.open(os.path.join(
            self.data_path, image_name)).convert('RGB')
        # print(os.path.join(self.data_path, image_name))
        img_class = self.gt_annotations[self.mode][image_name]
        label = np.zeros(self.num_classes)
        label[int(img_class) - 1] = 1
        label = torch.from_numpy(label)
        label = label.type(torch.FloatTensor)
        label_name = self.classes_labels[img_class]

        if self.transform:
            img = self.transform(img)

        return img, label, label_name


if __name__ == "__main__":
    dataset = CommonDataset(data_path="./data/context/", mode="train", split=0)
    dataset = CommonDataset(data_path="./data/context/", mode="test", split=1)
