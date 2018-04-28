from __future__ import absolute_import

import torch.nn as nn
from .base import Layer


class Input(Layer):
    """
    Input layer, always the first layer of any model
    """
    def __init__(self, insize, name=None):
        super(Input, self).__init__([insize], [insize], name)

    def forward(self, input):
        return input


class Dropout(Layer):
    """Dropoyt layer"""
    def __init__(self, dropout, name=None):
        """
        Initialize layer
        :param dropout: percentage of weights dropped
        :param name: name of current layer
        """
        super(Dropout, self).__init__([None], [None], name)
        self.kernel = nn.Dropout(dropout)


class Reshape(Layer):
    """Reshape the output"""
    def __init__(self, shape, name=None):
        """
        Initialize layer
        :param shape: target shape of the tensor
        :param name: name of current layer
        """
        super(Reshape, self).__init__([None], [None], name)
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)
