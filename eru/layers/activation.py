from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F
from .base import Layer


class Softmax(Layer):
    """Softmax Layer"""
    def __init__(self, insize, outsize):
        super(Softmax, self).__init__(insize, outsize, name=None)
        self.kernel = nn.LogSoftmax(dim=2)


class Sigmoid(Layer):
    """Sigmoid Layer"""
    def __init__(self, insize, outsize):
        super(Sigmoid, self).__init__(insize, outsize, name=None)
        self.kernel = nn.LogSigmoid()


class Activation(Layer):
    """Other Activation Layer"""
    def __init__(self, non_linearity_type, name=None):
        super(Activation, self).__init__([None], [None], name)
        if non_linearity_type == "softmax":
            self.kernel = nn.LogSoftmax(dim=2)
        else:
            self.kernel = getattr(F, non_linearity_type)
