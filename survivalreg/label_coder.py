from abc import ABC, abstractmethod
from typing import List, Any

import numpy as np
import torch

from .util.logger import get_logger

logger = get_logger(__name__)


class LabelCoderBaseClass(ABC):
    @abstractmethod
    def code(self, label) -> List[bool]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, code: List[bool]):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


class RankLabelCoder(LabelCoderBaseClass):
    def __init__(self, thresholds: List[Any]):
        self.n_classes = len(thresholds)
        self.thresholds = thresholds

    def code(self, label) -> List[bool]:
        arr = torch.FloatTensor([label > t for t in self.thresholds])
        return (arr.to(torch.float32) - 0.5) * 2

    def decode(self, code: List[float]):
        return code

    def __len__(self):
        return self.n_classes


class BinaryLabelCoder(LabelCoderBaseClass):
    def __init__(self):
        pass

    def code(self, label) -> List[bool]:
        if isinstance(label, list):
            label = torch.Tensor(label)
        if isinstance(label, torch.FloatTensor):
            label = label > 0

        code = (label.to(torch.float32) - 0.5) * 2
        # if len(code.shape) == 1:
        #     code = code.view(-1, 1)
        return code

    def decode(self, code: List[bool]):
        return code

    def __len__(self):
        return 1


class LabelCoder:
    def __init__(self) -> None:
        coders = []
        for k, v in self.__class__.__dict__.items():
            if isinstance(v, LabelCoderBaseClass):
                coders.append((k, v))
        self.digit_offset = [0]
        for i, (_, coder) in enumerate(coders):
            self.digit_offset.append(self.digit_offset[i] + len(coder))
        self.coders = coders

    def __len__(self):
        return sum([len(c) for _, c in self.coders])

    def __call__(self, *args, **kwargs):
        return self.code(*args, **kwargs)

    def code(self, label) -> torch.Tensor:
        # logger.info(label)
        collection = []
        for name, coder in self.coders:
            assert name in label
            collection.append(coder.code(label[name]))
        res = torch.stack(collection, dim=1)
        # logger.info(res.shape)
        return res

    def decode(self, code: torch.Tensor) -> dict:
        assert isinstance(code, torch.Tensor) or isinstance(code, np.ndarray)
        result = {}
        for i, (name, coder) in enumerate(self.coders):
            result[name] = code[..., self.digit_offset[i]:self.digit_offset[i + 1]]
        return result


if __name__ == '__main__':
    class TestCoder(LabelCoder):
        a = RankLabelCoder([0, 1, 2])
        b = BinaryLabelCoder()


    lc = TestCoder()
    print(lc.code({'a': 2, 'b': True}))
    print(len(lc))
    print(lc.digit_offset)
    print(lc.decode(lc.code({'a': 2, 'b': True})))
