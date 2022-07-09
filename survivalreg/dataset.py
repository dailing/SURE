from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from functools import cached_property
from random import Random
from typing import List, Tuple

Sample = namedtuple('Sample', ['sid', 'time', 'label'])

merged_data = namedtuple('merged_data', ['accindex', 'array'])


class SurvivalDataset(ABC):
    def __init__(self, testing=False, sample_seed=None) -> None:
        super().__init__()
        self._testing = testing
        self._rnd = Random(sample_seed)

    @abstractmethod
    def info(self, index: int) -> Sample:
        """
        :param index: index of the item
        :return: should be a instance of Sample and sorted by id
        :Raises: IndexError if index is out of range
        """
        raise NotImplementedError

    @abstractmethod
    def feature(self, index: int):
        """
        :param index: index of the item
        :return: sould return the feature as the input of models
        """
        raise NotImplementedError

    @cached_property
    def merged_data(self):
        data = defaultdict(list)
        index = 0
        while True:
            try:
                sample = self.info(index)
                assert isinstance(sample, Sample)
                data[sample.sid].append((sample, index))
                index += 1
            except IndexError:
                break
        # sort each set of samples by time
        for k, v in data.items():
            v.sort(key=lambda x: x[0].time)
        # make the data into a list of samples
        accumulated_index = []
        flat_data = []
        for v in data.values():
            accumulated_index.append(len(flat_data))
            flat_data.extend(v)
        accumulated_index.append(len(flat_data))
        return merged_data(accumulated_index, flat_data)

    def __len__(self):
        if self._testing:
            return len(self.merged_data.accindex) - 1
        else:
            return len(self.merged_data.accindex) - 1

    def _item_train(self, index):
        assert 0 <= index < len(self)
        records = self.merged_data.array[
                  self.merged_data.accindex[index]:
                  self.merged_data.accindex[index + 1]]
        # randomly sample tow records with replacement
        records = self._rnd.choices(records, k=2)
        return records

    def _item_test(self, index):
        """
        generate test samples. test samples are iteratioin of the whole dataset
        if the size of the dataset is odd, the last sample is duplicated
        """
        assert 0 <= index < len(self)
        sample1 = index
        sample2 = min(index + 1, len(self) - 1)
        return [self.merged_data[1][sample1],
                self.merged_data[1][sample2]]

    def _handle_paired_sample(self, sample_pair: List[Tuple[Sample, Sample]]):
        """
        generate the trainable data for the model given a pair of samples
        """
        sample1, index1 = sample_pair[0]
        sample2, index2 = sample_pair[1]
        feat1 = self.feature(index1)
        feat2 = self.feature(index2)
        label1 = sample1.label
        label2 = sample2.label
        dt = sample2.time - sample1.time
        return feat1, feat2, label1, label2, dt

    def __getitem__(self, index):
        if self._testing:
            item = self._item_test(index)
        else:
            item = self._item_train(index)
        return self._handle_paired_sample(item)
