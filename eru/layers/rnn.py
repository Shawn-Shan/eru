from __future__ import absolute_import
from torch.autograd import Variable
from .base import *


class GRU(Layer):
    """GRU layer"""

    def __init__(self, insize, hidden_size, name=None, nlayers=1, recurrent_dropout=0.1, return_sequence=False,
                 extract_hidden=False):
        """
        Initialize current layer
        :param insize: input size of current layer
        :param hidden_size: output/hidden size of current layer
        :param name: the given name of current layer
        :param nlayers: number of recurrent layers
        :param recurrent_dropout: recurrent dropout rate
        :param return_sequence: whether return the sequence output or only last time step output
        :param extract_hidden: whether return final hidden states
        """
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
        """
        Initialize the hidden states of the network
        """
        weight = next(self.parameters()).data
        self.hidden = Variable(weight.new(self.nlayers, self.batch_size, self.hidden_size).zero_().cuda())
        return self.hidden
