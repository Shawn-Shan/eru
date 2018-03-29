from __future__ import absolute_import
import random
import torch
from .utils import batchify
from torch.autograd import Variable


def get_next_word_batch(source, i, seq_len=35, evaluation=False, return_sequence=True):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = Variable(source[i:i + seq_len], volatile=evaluation).cuda()
    if return_sequence:
        target = Variable(source[i + 1:i + 1 + seq_len].view(-1)).cuda()
    else:
        target = Variable(source[i + seq_len].view(-1)).cuda()

    return data, target


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.null_token = '<NULL>'

    def encode_word2idx(self, word, enable_null=False):
        if word in self.word2idx:
            return self.word2idx[word]
        elif enable_null:
            return self.word2idx[self.null_token]
        else:
            raise IndexError('Word not in dictionary, if want to replace with Null Token call enable_null=True')

    def encode_idx2word(self, idx):
        return self.idx2word[idx]

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class simple_generator(object):
    def __init__(self, x, y, x_test=None, y_test=None, batch_size=128, input_categorical=True, output_categorical=True,
                 train_test_split=0.9):
        self.batch_size = batch_size
        self.train_test_split = train_test_split
        self.x_train, self.x_test = self.init_tensor(x, x_test, input_categorical)
        self.y_train, self.y_test = self.init_tensor(y, y_test, output_categorical)
        self.bound = len(self.y_train)

    def init_tensor(self, train_tensor, test_tensor, categorical):
        if test_tensor is None:
            total_len = len(train_tensor)
            split_point = int(total_len * self.train_test_split)
            test_tensor = train_tensor[split_point:]
            train_tensor = train_tensor[:split_point]
        if categorical:
            train_tensor = Variable(torch.cuda.LongTensor(train_tensor))
            test_tensor = Variable(torch.cuda.LongTensor(test_tensor))
        else:
            train_tensor = Variable(torch.cuda.FloatTensor(train_tensor))
            test_tensor = Variable(torch.cuda.FloatTensor(test_tensor))

        train_tensor = batchify(train_tensor, self.batch_size)
        test_tensor = batchify(test_tensor, self.batch_size)
        return train_tensor, test_tensor

    def generate(self, evaluate=False):
        # i = 0
        while 1:
            i = random.randrange(0, self.bound)
            if not evaluate:
                # Training Case
                yield self.x_train[i], self.y_train[i]
            else:
                # Evaluation Case
                yield self.x_test[i], self.y_test[i]
                # i += 1

    def __len__(self):
        return len(self.y_train)


class Onehot_Encoder(object):
    def __init__(self, possible_inputs):
        self.possible_inputs = possible_inputs
        self.word2idx = dict((w, i) for i, w in enumerate(possible_inputs))
        self.idx2word = dict((i, w) for i, w in enumerate(possible_inputs))

    def encode(self, input):
        res = Variable(torch.cuda.FloatTensor(len(input), len(self.possible_inputs)).zero_())

        for i, value in enumerate(input):
            res[i, self.word2idx[value]] = 1
        return res

    def encode_batch(self, batch_input):

        for i, b in enumerate(batch_input):
            batch_input[i] = self.encode(b)

        return torch.stack(batch_input, dim=1)

    def forward(self, input):
        return self.encode_batch(input)
