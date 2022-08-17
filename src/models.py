import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F


class CustomFrozenModel(nn.Module):

    def __init__(self, num_classes=20):
        super(CustomFrozenModel, self).__init__()
        self.num_classes = num_classes
        self.base_model = self.get_custom_model(
            torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True), freeze=True)
        self.act = nn.LeakyReLU()
        self.opt_layer = nn.Linear(1000, self.num_classes)

    def get_custom_model(self, model, freeze=True):
        if not freeze:
            return model
        else:
            ct = 0
            for child in model.children():
                if ct < 6:
                    for param in child.parameters():
                        param.requires_grad_(False)
                else:
                    for param in child.parameters():
                        param.requires_grad_(True)
                ct += 1
            return model

            """
            for idx, (name, layer) in enumerate(model.named_modules()):
                param.requires_grad = False
                if isinstance(layer, torch.nn.Conv2d) and idx<34:
                    layer.requires_grad_(False)
            else:
                layer.requires_grad_(True)"""

    def sanity_check(self):
        for idx, (name, layer) in enumerate(self.base_model.named_modules()):
            if isinstance(layer, torch.nn.Conv2d):
                print(name, "->", layer.weight.requires_grad)

    def forward(self, x):
        ftrs = self.base_model(x)
        act_ftrs = self.act(ftrs)
        logits = self.opt_layer(act_ftrs)
        return logits


class CustomSimpleClassifier(nn.Module):
    def __init__(self):
        super(CustomSimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(44944, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Encoder_block(nn.Module):
    def __init__(self, input_kernel_nums, output_kernel_nums, kernel_size):
        super(Encoder_block, self).__init__()
        self.conv = nn.Conv2d(
            input_kernel_nums, output_kernel_nums, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(100)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.bn(x)
        return x


class Encoder_block(nn.Module):
    def __init__(self, input_kernel_nums, output_kernel_nums, kernel_size):
        super(Encoder_block, self).__init__()
        self.conv = nn.Conv2d(
            input_kernel_nums, output_kernel_nums, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(output_kernel_nums)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        print(x.shape)
        x = self.bn(x)
        return x


class TaskModel(nn.Module):
    def __init__(self, num_classes=20):
        super(TaskModel, self).__init__()
        self.cnn_block1 = Encoder_block(3, 8, 7)
        self.cnn_block2 = Encoder_block(8, 16, 5)
        self.cnn_block3 = Encoder_block(16, 32, 5)
        self.cnn_block4 = Encoder_block(32, 64, 3)
        self.cnn_block5 = Encoder_block(64, 128, 3)
        self.cnn_block6 = Encoder_block(128, 256, 3)
        self.fc1 = nn.Linear(256, num_classes)
        self.drop3 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.3)
        self.drop1 = nn.Dropout(0.4)

    def forward(self, x):
        x = self.cnn_block1(x)
        x = self.drop1(x)
        x = self.cnn_block2(x)
        x = self.drop2(x)
        x = self.cnn_block3(x)
        x = self.drop3(x)
        x = self.cnn_block4(x)
        x = self.cnn_block5(x)
        x = self.cnn_block6(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        print(x.shape)
        return x
