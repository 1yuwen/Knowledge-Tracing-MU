import abc
import torch
import os.path as osp
from dataloader.data_utils import *

from \
    utils import (
    ensure_path,
    Averager, Timer, count_acc,
)


class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        self.args = set_up_datasets(self.args)
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.trlog = {}
        self.trlog['train_loss'] = []
        self.trlog['test_loss'] = []
        self.trlog['train_fine_acc'] = []
        self.trlog['train_coarse_acc'] = []
        self.trlog['test_coarse_acc'] = []
        self.trlog['test_fine_acc'] = []
        self.trlog['fine_max_acc_epoch'] = 0
        self.trlog['fine_max_acc'] = 0.0
        self.trlog['fine_min_acc_epoch'] = 0
        self.trlog['fine_min_acc'] = 100.0
        self.trlog['coarse_max_acc_epoch'] = 0
        self.trlog['coarse_max_acc'] = 0.0
        self.trlog['coarse_min_acc_epoch'] = 0
        self.trlog['coarse_min_acc'] = 100.0


    @abc.abstractmethod
    def train(self):
        pass