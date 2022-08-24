import clip

import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F


class CustomModel(nn.Module):

    def __init__(self, base_model=None, num_classes=20):
        super(CustomModel, self).__init__()
        self.num_classes = num_classes
        self.base_model = torch.hub.load(
            'pytorch/vision:v0.10.0', 'resnet18', pretrained=True)  # vgg19_bn
        # res18 -> 512, res152 -> 2048
        self.base_model.fc = nn.Linear(512, self.num_classes)

    def forward(self, x):
        logits = self.base_model(x)
        return logits


class ClipModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        with open('data/context/classes.json', 'r') as fp:
            classes_labels = json.load(fp)
        self.class_labels = list(classes_labels.values())
        self.text_embeds = clip.tokenize(self.class_labels).to(device)
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

    def forward(self, image, label_name, return_ftrs=False, mode='val'):

        # image = self.preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
        # text = clip.tokenize(list(label_name)).to(device)
        # print(label_name)
        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(self.text_embeds)

        logits_per_image, logits_per_text = self.model(image, self.text_embeds)

        return logits_per_image, image_features, text_features
