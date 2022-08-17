import os
import cv2
import random
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torchvision.transforms as transforms


class BottleDataset(Dataset):
    def __init__(self, data_path, split="train"):
        self.split = split
        self.data_path = data_path
        self.split_size = 0.2
        if self.split == "train":
            self.split_size = 1 - self.split_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        # getting all images
        self.img_names = os.listdir(self.data_path)
        # selecting first k of shuffled
        k = int(len(self.img_names)*self.split_size)
        random.shuffle(self.img_names)
        self.data = self.img_names[:k]
        self.visualize()

    def visualize(self, w=20, h=20):
        data_dist = {}
        for name in self.data:
            k = int(name.split("-")[0])-1
            if k not in data_dist:
                data_dist[k] = 0
            data_dist[k] += 1
        print("\n\n")

        fig = plt.figure(figsize=(8, 8))
        columns = 4
        rows = 5
        for i in range(1, columns*rows + 1):
            img = cv2.imread(os.path.join(self.data_path, self.data[i]))
            fig.add_subplot(rows, columns, i)
            plt.title(int(self.data[i].split("-")[0])-1)
            plt.imshow(img)
        plt.show()
        print("\n\n")
        data_dist = dict(sorted(data_dist.items()))
        plt.bar(range(len(data_dist)), list(
            data_dist.values()), align='center')
        plt.xticks(range(len(data_dist)), list(data_dist.keys()))
        plt.show()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_path, self.data[idx]))
        label = int(self.data[idx].split("-")[0])-1

        if self.transform:
            img = self.transform(img)

        return img, label
