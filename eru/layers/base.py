from __future__ import absolute_import
import torch.nn as nn


def create_layer_name(layer, idx):
    name = layer.__class__.__name__
    return name + "_" + str(idx)


class Layer(nn.Module):
    """The base class of layer
    """
    def __init__(self, inshape, outshape, name):
        """
        Initialize the layer with inshape, outshape, name)
        :param inshape: a tuple of the input shape of current layer
        :param outshape: a tuple of the output shape of current layer
        :param name: the name of current layer
        """
        super(Layer, self).__init__()
        
        self.batch_size = None
        self.inshape = [self.batch_size, *inshape]
        self.outshape = [self.batch_size, *outshape]
        self.name = name
        self.in_bound_layers = []
        self.out_bound_layers = []
        
        self.kernel = None
        self.require_hidden = False
        self.hidden = None

    def forward(self, input):
        """
        Feed input through the layer
        :param input: input data of the layer
        :return: output data of the layer
        """
        return self.kernel(input)
    
    def update_batch_size(self, bsz):
        """
        Update the batch size of current layer
        :param bsz:
        :return:
        """
        self.batch_size = bsz
        self.inshape[0] = bsz
        self.outshape[0] = bsz

    def __call__(self, layer):
        """
        Add current layer as the children of given layer to construct the model
        :param layer: the parent layer. Current layer takes the parents layer's output as input
        :return: current layer
        """
        self.cuda()
        self.in_bound_layers.append(layer)
        layer.out_bound_layers.append(self)
        return self