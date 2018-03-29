from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F
from .base import Layer

class Softmax(Layer):
    def __init__(self, insize, outsize):
        super(Softmax, self).__init__(insize, outsize)
        self.kernel = nn.LogSoftmax(dim=2)


class Sigmoid(Layer):
    def __init__(self, insize, outsize):
        super(Sigmoid, self).__init__(insize, outsize)
        self.kernel = nn.LogSigmoid(dim=2)

        
class Activation(Layer):
    def __init__(self, non_linearity_type, name=None):
        super(Activation, self).__init__([None], [None], name)
        self.kernel = getattr(F, non_linearity_type)