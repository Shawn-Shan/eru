from __future__ import absolute_import

import torch.nn as nn
from torch.nn import init
from .base import Layer


class Dense(Layer):
    """Basic linear layer"""
    def __init__(self, insize, outsize, name=None):
        """
        Initialize current layer
        :param insize: input size of current layer
        :param outsize: output size of current layer
        :param name: the given name of current layer
        """
        super(Dense, self).__init__([insize], [outsize], name)
        self.out_shape = outsize
        self.kernel = nn.Linear(insize, self.out_shape)
        self.in_shape = insize
        init.xavier_uniform(self.kernel.weight)