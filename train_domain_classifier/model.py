import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np


def to_categorical(y, num_classes):
    return torch.tensor(np.eye(num_classes, dtype='float')[y.int().cpu()]).cuda().float().view(y.size(0), -1)


class INet(nn.Module):
    def __init__(self):
        super(INet, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.mlp = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(in_features=2048, out_features=128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, 3)
                    )

        # self.intention_mlp = nn.Sequential(
        #     nn.Linear(3, 64),
        #     nn.ReLU(),
        #     nn.Dropout(0.7)
        # )

    def forward(self, img):
        x = self.backbone(img)
        # intention_feat = to_categorical(intention, 3)
        # intention_feat = self.intention_mlp(intention_feat)
        # x = torch.cat((feature, intention_feat), dim=1)
        x = self.mlp(x)
        m = nn.Softmax(dim=1)
        x = m(x)
        return x