from __future__ import absolute_import
import torch
from torch.autograd import Variable
import random
import numpy as np


class simple_loader(object):
    """
    A default data loader and used when Model API get normal x, y pairs as inputs
    """

    def __init__(self, x, y, x_test=None, y_test=None, batch_size=32, input_categorical=True, output_categorical=True,
                 train_test_split=None, shuffle=True):
        """
        Initialize the generator with input data
        :param x: training data x
        :param y: target training y
        :param x_test: testing data x
        :param y_test: target testing y
        :param batch_size: output batch size
        :param input_categorical: whether the input is categorical
        :param output_categorical: whether the output is categorical
        :param train_test_split: percentage of training data use as test
        :param shuffle: whether to shuffle the training set
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_only = False

        if train_test_split is None and x_test is None:
            self.train_only = True
        self.train_test_split = train_test_split

        self.x_train, self.x_test = self.init_tensor(x, x_test, input_categorical)
        self.y_train, self.y_test = self.init_tensor(y, y_test, output_categorical)
        self.bound = len(self.y_train)

    def init_tensor(self, train_tensor, test_tensor, categorical):
        """
        Convert input data to pytorch tensor/variable class
        :param train_tensor: train tensor input
        :param test_tensor: testing tensor input
        :param categorical: whether current tensor is categorical
        :return: training tensor as pytorch variable, testing tensor as pytorch variable
        """
        if isinstance(train_tensor, list):
            try:
                train_tensor = np.array(train_tensor, dtype="float32")
            except ValueError:
                raise ValueError("Input list have bad format")

        if isinstance(test_tensor, list):
            try:
                train_tensor = np.array(train_tensor, dtype="float32")
            except ValueError:
                raise ValueError("Input list have bad format")

        if not self.train_only:
            if test_tensor is None:
                total_len = len(train_tensor)
                split_point = int(total_len * self.train_test_split)
                test_tensor = train_tensor[split_point:]
                train_tensor = train_tensor[:split_point]

        if categorical:
            train_tensor = Variable(torch.cuda.LongTensor(train_tensor))
            if not self.train_only:
                test_tensor = Variable(torch.cuda.LongTensor(test_tensor))
        else:
            train_tensor = Variable(torch.cuda.FloatTensor(train_tensor))
            if not self.train_only:
                test_tensor = Variable(torch.cuda.FloatTensor(test_tensor))

        return train_tensor, test_tensor

    def generate(self):
        """
        Main data generator
        :return: yield x, y batch at each time step
        """
        batch_x = []
        batch_y = []
        while 1:
            i = random.randrange(0, self.bound)
            cur_x, cur_y = self.x_train[i], self.y_train[i]
            batch_x.append(cur_x)
            batch_y.append(cur_y)

            if len(batch_y) == self.batch_size:
                batch_x = torch.stack(batch_x, dim=0)
                batch_y = torch.stack(batch_y, dim=0)
                yield batch_x, batch_y
                batch_x = []
                batch_y = []
            i += 1

    def __len__(self):
        """
        Get the length of the training steps
        :return: Number of steps in on epochs
        """
        return len(self.y_train)
