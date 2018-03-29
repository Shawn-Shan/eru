from __future__ import absolute_import

import torch.nn as nn
from torch.nn import init
import numpy as np
import torch
from .base import Layer

class Embedding(Layer):
    def __init__(self, insize, outsize, name=None, pretrain=None, dictionary=None):
        super(Embedding, self).__init__([insize], [insize, outsize], name)
        self.kernel = nn.Embedding(insize, outsize)
        init.xavier_uniform(self.kernel.weight)

        if pretrain is not None:
            assert(dictionary is not None)
            print("Loading pretrained embedding from {}".format(pretrain))
            embeddings_index = {}
            with open(pretrain) as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
            emb_layer = torch.cuda.FloatTensor(insize, outsize)
            
            c = 0
            for w in dictionary.word2idx.keys():
                idx = dictionary.word2idx[w]
                if w in embeddings_index:
                    cur_emb = embeddings_index[w]
                else:
                    c += 1
                    continue
                cur_emb = torch.cuda.LongTensor(cur_emb)
                emb_layer[idx, :] = cur_emb
            print("{} out of {} words not in Embedding".format(c, insize))
            self.kernel.weight = nn.Parameter(emb_layer)