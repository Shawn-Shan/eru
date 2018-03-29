from __future__ import absolute_import

import torch.nn as nn


def create_layer_name(layer, idx):
    name = layer.__class__.__name__
    return name + "_" + str(idx)
    

class Layer(nn.Module):
    def __init__(self, inshape, outshape, name):
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
        return self.kernel(input)
    
    def update_batch_size(self, bsz):
        self.batch_size = bsz
        self.inshape[0] = bsz
        self.outshape[0] = bsz

    def __call__(self, layer):
        self.in_bound_layers.append(layer)
        layer.out_bound_layers.append(self)
        return self