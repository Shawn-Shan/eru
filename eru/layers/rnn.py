from __future__ import absolute_import

import torch.nn as nn
from torch.autograd import Variable
from .base import *

class GRU(Layer):
    def __init__(self, insize, hidden_size, name=None, nlayers=1, recurrent_dropout=0.1, return_sequence=True, extract_hidden=False):
        self.insize = insize
        if return_sequence:
            inshape = [None, insize]
            outshape = [None, hidden_size]
        else:
            inshape = [insize]
            outshape = [hidden_size]
        
        super(GRU, self).__init__(inshape, outshape, name)
        self.hidden_size = hidden_size
        self.extract_hidden = extract_hidden
        self.nlayers = nlayers
        self.recurrent_dropout = recurrent_dropout
        self.kernel = nn.GRU(insize, hidden_size, self.nlayers, dropout=self.recurrent_dropout)
        self.return_sequence = return_sequence
        self.require_hidden = True

    def forward(self, input):
        input = input.view(-1, self.batch_size, self.insize)
        self.hidden = self.init_hidden()
        output, hidden = self.kernel(input, self.hidden)

        if not self.return_sequence:
            output = output[-1:]

        if self.extract_hidden:
            return hidden
        else:
            return output.view(self.batch_size, -1, self.hidden_size)

    def init_hidden(self):
        weight = next(self.parameters()).data
        self.hidden = Variable(weight.new(self.nlayers, self.batch_size, self.hidden_size).zero_().cuda())
