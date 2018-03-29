from __future__ import absolute_import

import torch.nn as nn
from torch.autograd import Variable
from .base import Layer

class Input(Layer):
    def __init__(self, insize, name=None):
        super(Input, self).__init__([insize], [insize], name)
        
    def forward(self, input):
        return input
    
    
class Dropout(Layer):
    def __init__(self, dropout, name=None):
        super(Dropout, self).__init__([None], [None], name)
        self.kernel = nn.Dropout(dropout)
        
class Reshape(Layer):
    def __init__(self, shape, name=None):
        super(Reshape, self).__init__([None], [None], name)
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)