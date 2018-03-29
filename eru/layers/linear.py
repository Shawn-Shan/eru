from __future__ import absolute_import

import torch.nn as nn
from torch.nn import init
from .base import Layer

class Linear(Layer):
    def __init__(self, insize, outsize, name=None):
        super(Linear, self).__init__([insize], [outsize], name)
        self.out_shape = outsize
        self.kernel = nn.Linear(insize, self.out_shape)
        self.in_shape = insize
        init.xavier_uniform(self.kernel.weight)