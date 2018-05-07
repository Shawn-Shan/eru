from __future__ import absolute_import

import torch.nn as nn
from torch.autograd import Variable
from .base import *


class conv1d(Layer):
	def __init__(self,in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias):
		'''
		Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        '''

       	super(conv1d,self).__init__([in_channels],[out_channels],name)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.kernel = nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias)
        init.xavier_uniform(self.kernel.weight)