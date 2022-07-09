import abc
import json
import os
import pickle
import time
from ast import Dict
from collections import defaultdict
from functools import cached_property

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .label_coder import LabelCoder
from .model import ModelProgression
from .util.config import Config, Parser


class TrainerConfig(Config):
    debug = Parser('debug', 'False',
                   lambda x: not x.lower().startswith('f'), 'debug mode ')
    load_pretrain = Parser('load_pretrain', None, str, 'load pretrained model')
    batch_size = Parser('batch_size', 128, int, 'batch size')
    epochs = Parser('epochs', 100, int, 'number of max epochs to train')
    image_size = Parser('image_size', 224, int, 'image size')
    lr = Parser('lr', 0.001, float, 'learning rate')
    device = Parser('device', 'cuda:0', str, 'device')
    num_workers = Parser('num_workers', 4, int, 'number of workers')
    model = Parser('model', 'convnext_tiny', str, 'backbone model')


class Trainer():
    cfg = TrainerConfig
    label_coder: LabelCoder = None

    def __init__(self) -> None:
        print(self.__class__)
        tt = time.gmtime()
        self.running_uuid = f'{tt.tm_year}_{tt.tm_mon:02d}_{tt.tm_mday:02d}-{tt.tm_hour:02d}_{tt.tm_min:02d}_{tt.tm_sec:02d}'
        print('running_uuid', self.running_uuid)
        print(self.cfg)
        self.epoch = None

    def _get_cfg_recursive(self, cls=None):
        if cls is None:
            cls = self.__class__
        parent_cfg = Config()
        for parent in cls.__bases__:
            parent_cfg.__dict__.update(
                self._get_cfg_recursive(parent).__dict__)
        if hasattr(cls, '_cfg'):
            cfg = cls._cfg()
            parent_cfg.__dict__.update(cfg.__dict__)
        return parent_cfg

    @cached_property
    def logger_dir(self):
        logger_dir = f'logs/{self.running_uuid}'
        if self.cfg.debug:
            logger_dir = f'logs/debug_{self.running_uuid}'
            print(f'logger_dir: {logger_dir}')
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)
        return logger_dir

    @cached_property
    def training_log(self):
        training_log = open(os.path.join(
            self.logger_dir, f'training_log.txt'), 'w')
        return training_log

    @cached_property
    def model(self):
        model = ModelProgression(
            backbone=self.cfg.model,
            output_size=len(self.label_coder))
        if self.cfg.load_pretrain:
            model.load_state_dict(torch.load(
                self.cfg.load_pretrain, map_location='cpu'))
            print(f'load pretrain: {self.cfg.load_pretrain}')
        return model

    @cached_property
    def train_dataset(self)->Dataset:
        raise NotImplementedError

    @cached_property
    def test_dataset(self) -> Dataset:
        raise NotImplementedError

    @cached_property
    def train_loader(self) -> DataLoader:
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            persistent_workers=True
        )
        return loader

    @cached_property
    def test_loader(self) -> DataLoader:
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=True
        )
        return loader

    @cached_property
    def device(self):
        if self.cfg.device.startswith('cuda') and not torch.cuda.is_available():
            print('cuda is not available, using CPU mode')
            self.cfg.device = 'cpu'
        device = torch.device(self.cfg.device)
        return device

    @cached_property
    def optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        return optimizer

    @cached_property
    def criterion(self):
        return nn.CrossEntropyLoss()

    @cached_property
    def scheduler(self):
        sch = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.1)
        self.optimizer.zero_grad()
        self.optimizer.step()
        return sch

    @abc.abstractmethod
    def batch(self, epoch, ibatch, data) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def matrix(self, epoch, data) -> dict:
        raise NotImplementedError

    def train(self):
        print(self.cfg, file=self.training_log)
        for epoch in range(self.cfg.epochs):
            self.epoch = epoch
            self.model.train()
            self.model.to(self.device)
            outputs = defaultdict(list)
            for i_batch, batch_data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.batch(epoch, i_batch, batch_data)
                loss = output.pop('loss')
                loss.backward()
                self.optimizer.step()

                for k, v in output.items():
                    outputs[k].append(v.detach().cpu())
                print(f'training {self.running_uuid} epoch:{epoch}/{self.cfg.epochs} '
                      f'batch {i_batch}/{len(self.train_loader)} {float(loss):.3f}', end='\r')
                print(json.dumps(dict(
                    type='train',
                    epoch=epoch,
                    ibatch=i_batch,
                    loss=float(loss),
                )), file=self.training_log)
                self.training_log.flush()
                if self.cfg.debug and i_batch > 2:
                    break
            self.scheduler.step()
            torch.save(self.model.state_dict(), os.path.join(
                self.logger_dir, f'model_{epoch:03d}.pth'))
            # calculate matrix training
            collected = {}
            for k, v in outputs.items():
                if len(v[0].shape) == 0:
                    collected[k] = torch.stack(v)
                else:
                    collected[k] = torch.cat(v)
            metrix = self.matrix(epoch=self.epoch, data=collected)
            metrix.update(dict(
                type='train matrix',
                epoch=self.epoch,
            ))
            print(json.dumps(metrix), file=self.training_log)
            self.training_log.flush()
            print('epoch train mat ', self.epoch, end=' ')
            for k, v in metrix.items():
                print(f'{k}: {v}', end=' ')
            print()

            self.test()
            if self.cfg.debug:
                break

    def test(self):
        self.model.eval()
        self.model.to(self.device)
        outputs = defaultdict(list)
        with torch.no_grad():
            for ibatch, data in enumerate(self.test_loader):
                output = self.batch(epoch=self.epoch, ibatch=-1, data=data)
                for k, v in output.items():
                    outputs[k].append(v.detach().cpu())
                print(
                    f'testing{self.running_uuid} epoch:{self.epoch}/'
                    f'{self.cfg.epochs} batch {ibatch}/{len(self.test_loader)}',
                    end=' \r')
                if self.cfg.debug and ibatch > 2:
                    break
        collected = {}
        for k, v in outputs.items():
            if len(v[0].shape) == 0:
                collected[k] = torch.stack(v)
            else:
                collected[k] = torch.cat(v)
        metrix = self.matrix(epoch=self.epoch, data=collected)
        metrix.update(dict(
            type='test matrix',
            epoch=self.epoch,
        ))
        print(json.dumps(metrix), file=self.training_log)
        self.training_log.flush()
        print('epoch', self.epoch, end=' ')
        for k, v in metrix.items():
            print(f'{k}: {v}', end=' ')
        print()
        pickle.dump(collected, open(os.path.join(
            self.logger_dir, f'preds_{self.epoch}.pkl'), 'wb'))
