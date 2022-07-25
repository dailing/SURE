import os.path
from functools import cached_property
from typing import List

import cv2
import pandas as pd

from .dataset import SurvivalDataset, Sample


class CSVSurvivalDataset(SurvivalDataset):
    def __init__(self,
                 filename,
                 testing=False,
                 image_key='image',
                 id_key='id',
                 label_keys: List = None,
                 time_key='time',
                 transform=None,
                 image_size=224,
                 root=None,
                 **kwargs):
        super().__init__(testing=testing, **kwargs)
        self.testing = testing
        self.image_key = image_key
        self.id_key = id_key
        if isinstance(label_keys, str):
            label_keys = [label_keys]
        self.label_keys = label_keys
        self.time_key = time_key
        self.filename = filename
        self.image_size = image_size
        self.root = root
        if transform is not None:
            # noinspection PyPropertyAccess
            self.transform = transform

    def info(self, index: int) -> Sample:
        row = self.pd_file.iloc[index]
        sid = row[self.id_key]
        etime = row[self.time_key]
        labels = {k: row[k] for k in self.label_keys}
        return Sample(sid, etime, labels)

    def feature(self, index: int):
        image = self.pd_file.iloc[index][self.image_key]
        if self.root is not None:
            image = os.path.join(self.root, image)
        assert os.path.exists(image)
        image = cv2.imread(image)
        return self.transform(image=image)['image']

    @cached_property
    def transform(self):
        import albumentations as aug
        import albumentations.pytorch as aug_torch
        if self.testing:
            return aug.Compose([
                aug.SmallestMaxSize(max_size=self.image_size, always_apply=True),
                aug.CenterCrop(self.image_size, self.image_size, always_apply=True),
                aug.ToFloat(always_apply=True),
                aug_torch.ToTensorV2(),
            ])
        else:
            return aug.Compose([
                aug.SmallestMaxSize(max_size=self.image_size, always_apply=True),
                aug.CenterCrop(self.image_size, self.image_size, always_apply=True),
                aug.Flip(p=0.5),
                aug.ImageCompression(quality_lower=10, quality_upper=80, p=0.2),
                aug.MedianBlur(p=0.3),
                aug.RandomBrightnessContrast(p=0.5),
                aug.RandomGamma(p=0.2),
                aug.GaussNoise(p=0.2),
                aug.Rotate(border_mode=cv2.BORDER_CONSTANT, value=0, p=0.7, limit=45),
                aug.ToFloat(always_apply=True),
                aug_torch.ToTensorV2(),
            ])

    @cached_property
    def pd_file(self):
        if isinstance(self.filename, pd.DataFrame):
            return self.filename
        return pd.read_csv(self.filename)
