from functools import cached_property

import torch
from torch import nn
from torchvision.datasets.mnist import MNIST

from survivalreg.dataset import SurvivalDataset, Sample
from survivalreg.label_coder import LabelCoder, BinaryLabelCoder
from survivalreg.tmloss import SURELoss
from survivalreg.trainer import Trainer
import pickle


class DummyData(SurvivalDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mnist = MNIST('.output', download=True, train=not self._testing)

        label = mnist.targets
        data = mnist.data
        # print(sample_t1)
        self.dataset = [
            Sample(i // 2, label[i], dict(label=label[i] > 5))
            for i in range(len(label))
        ]
        self.label = label
        self.data = data

    def info(self, index: int) -> Sample:
        return self.dataset[index]

    def feature(self, index: int):
        return torch.unsqueeze(self.data[index], 0) / 255


class DeepSurModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.AvgPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.AvgPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc3(x)
        return x


class LabelCoder(LabelCoder):
    label = BinaryLabelCoder()


class TrainSpiral(Trainer):
    label_coder = LabelCoder

    @cached_property
    def model(self):
        return DeepSurModel()

    @property
    def train_dataset(self):
        return DummyData()

    @property
    def test_dataset(self):
        return DummyData(testing=True)

    @cached_property
    def criterion(self):
        return SURELoss(gama=2)



if __name__ == '__main__':
    trainer = TrainSpiral()
    trainer.train()

    # mm = pickle.load(open('logs/TrainSpiral_last/preds_50.pkl', 'rb'))