import os
import random
from functools import cached_property

import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader

from survivalreg.dataset import SurvivalDataset, Sample
from survivalreg.label_coder import BinaryLabelCoder, LabelCoder
from survivalreg.trainer import Trainer


class DummyData(SurvivalDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self._testing:
            sample_t1 = np.linspace(-15, 0, 500)
            sample_t2 = np.linspace(0, 15, 500)
        else:
            sample_t1 = (np.random.rand(30000) - 0.5) * 30
            sample_t2 = sample_t1 + (np.random.rand(30000) - 0.5 + 4)
        # print(sample_t1)
        self.dataset = [
            Sample(i, sample_t1[i], dict(c=sample_t1[i] > 0))
            for i in range(len(sample_t1))
        ]
        self.dataset.extend([
            Sample(i, sample_t2[i], dict(c=sample_t2[i] > 0))
            for i in range(len(sample_t1))
        ])
        # print(self.dataset)

    def info(self, index: int) -> Sample:
        return self.dataset[index]

    def feature(self, index: int):
        theta = self.dataset[index].time
        x = theta * np.cos(np.abs(theta))
        y = theta * np.sin(np.abs(theta))
        if not self._testing:
            x += (random.random() - 0.5) * 0.2
            y += (random.random() - 0.5) * 0.2
        return np.array([x, y])


class LabelCoder(LabelCoder):
    c = BinaryLabelCoder()


class DeepSurModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        # print(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        # print(x.shape)
        return x


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


if __name__ == '__main__':
    # dd = DummyData(testing=True)
    # print(dd[900])
    # print(len(dd))
    # print(dd.merged_data)

    # b = B()

    # os.environ['debug'] = 'True'
    os.environ['num_workers'] = '4'
    os.environ['lr'] = '0.05'
    os.environ['batch_size'] = '1024'
    os.environ['device'] = 'cpu'
    trainer = TrainSpiral()
    trainer.train()

    # loader = DataLoader(
    #     DummyData(), batch_size=5, shuffle=True, num_workers=2
    # )
    # print(next(iter(loader)))