from eru.utils import init_gpu
from eru.model import Model
from eru.layers import GRU, Dense, Input, Activation
import random
import torch
from urllib.request import urlopen
from torch.autograd import Variable

init_gpu(3)


url = "https://stuff.mit.edu/afs/sipb/contrib/pi/pi-billion.txt"
html = urlopen(url).read()
pi = html.decode()
pi = "3" + pi[2:]


class Onehot(object):
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

    def cuda(self):
        return self


possible_inputs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
possible_inputs = [str(x) for x in possible_inputs]

one_hot_encode = Onehot(possible_inputs)
batch_size = 128


class Gene(object):
    def __init__(self, data):
        self.data = data

    def generate(self):
        l = len(self.data)
        batch_x = []
        batch_y = []
        c = 0
        while 1:
            i = random.randrange(0, l - 50)
            x = self.data[i:i + 35]
            y = self.data[i + 35]
            y = [int(i) for i in y]
            batch_x.append(x)
            batch_y.append(y)
            c += 1
            if c == batch_size:
                batch_x = one_hot_encode.forward(batch_x)
                batch_y = Variable(torch.cuda.LongTensor(batch_y)).view(-1)
                yield batch_x, batch_y
                batch_x = []
                batch_y = []
                c = 0


input = Input(10)
x = GRU(10, 128, return_sequence=False)(input)
x = Dense(128, 10)(x)
x = Activation("softmax")(x)
cur_model = Model(input, x)

cur_model.compile("adam", "crossentropy", metrics=['loss', 'acc'])
g = Gene(pi)
cur_model.fit_generator(g, batch_size=128, epochs=1, train_length=1000)
