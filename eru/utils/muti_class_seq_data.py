from __future__ import absolute_import
from .utils import *
from .data import *
from collections import Counter
import pickle


class MutiClassSequensial_Corpus(object):
    def __init__(self, path_dictionary=[], load_dictionary=False, keep_rate=0.995):
        self.dictionary = Dictionary()
        self.path_dictionary = path_dictionary
        self.keep_rate = keep_rate
        if not load_dictionary:
            self.update_dictionary()
        else:
            print("Load Dictionary")
            if isinstance(load_dictionary, str):
                corpus = pickle.load(open(load_dictionary, "rb"))
                self.dictionary = corpus.dictionary
            else:
                self.dictionary = load_dictionary

        self.content = {}

        for cur_dict in self.path_dictionary:
            sub_dictionary, label = cur_dict
            self.content[label] = {}
            cur_train_txt = os.path.join(sub_dictionary, 'train.txt')
            cur_test_txt = os.path.join(sub_dictionary, 'test.txt')

            if os.path.exists(cur_train_txt):
                cur_train = self.tokenize(cur_train_txt)
                self.content[label]["train"] = cur_train

            if os.path.exists(cur_test_txt):
                cur_test = self.tokenize(cur_test_txt)
                self.content[label]["test"] = cur_test

    def append_sub_corpus(self, path):
        with open(os.path.join(path, 'train.txt'), 'r') as f:
            for line in f:
                words = line.split()
                for word in words:
                    if word in self.total_count:
                        self.total_count[word] += 1
                    else:
                        self.total_count[word] = 1

    def update_dictionary(self, addition_words=[]):
        self.total_count = {}

        # Add words to the dictionary
        for cur_dict in self.path_dictionary:
            sub_dictionary, label = cur_dict
            print(sub_dictionary)
            self.append_sub_corpus(sub_dictionary)

        self.total_count = Counter(self.total_count)
        total_len = sum(self.total_count.values())
        for max_count in range(0, 1000000, 1000):
            new_dict = dict(self.total_count.most_common(max_count))
            presentage = sum(new_dict.values()) / total_len
            if presentage > self.keep_rate:
                print("Total {} words, Keep top {} words".format(str(len(self.total_count)), str(max_count)))
                break

        # Add to dictionary
        for key in new_dict.keys():
            self.dictionary.add_word(key)

        for ele in addition_words:
            self.dictionary.add_word(ele)

        self.dictionary.add_word('<NULL>')

    def tokenize(self, path):
        """Tokenizes a text file."""
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split()
                tokens += len(words)

        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split()
                for word in words:
                    if word not in self.dictionary.word2idx:
                        word = "<NULL>"
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


class MutiClassSequensial_Loader(object):
    def __init__(self, corpus, batch_size, evaluation=False, seq_len=35, output_categorical=True):
        self.batch_size = batch_size
        self.evaluation = evaluation
        self.seq_len = seq_len
        self.output_categorical = output_categorical
        self.corpus = corpus
        self.data = self.corpus.content
        self.data_dictionary = {}
        index = 0
        for key, value in self.data.items():
            label = key
            if evaluation:
                data = self.data[label]['test']
            else:
                data = self.data[label]['train']

            data = batchify(data, self.batch_size)

            self.data_dictionary[index] = {}
            self.data_dictionary[index]["label"] = label
            self.data_dictionary[index]["data"] = data
            self.data_dictionary[index]["bound"] = len(data) - 1 - self.seq_len
            index += 1
        self.index = index

    def generate(self):
        if self.evaluation:
            index = 0
            i = 0
            while 1:
                select = self.data_dictionary[index]
                source = select["data"]
                label = select["label"]
                bound = select["bound"]
                while i < bound:
                    batch_x = Variable(source[i:i + self.seq_len]).cuda()
                    if self.output_categorical:
                        target = Variable(torch.cuda.LongTensor([label] * self.batch_size)).view(1, -1)
                    else:
                        target = Variable(torch.cuda.FloatTensor([label] * self.batch_size)).view(1, -1)
                    yield batch_x, target
                    i += 1
                index += 1
                i = 0
        else:
            while 1:
                pointer = random.choice(range(self.index))
                selected = self.data_dictionary[pointer]
                i = random.randrange(0, selected["bound"])
                source = selected["data"]
                label = selected["label"]
                batch_x = Variable(source[i:i + self.seq_len]).cuda()
                if self.output_categorical:
                    target = Variable(torch.cuda.LongTensor([label] * self.batch_size)).view(1, -1)
                else:
                    target = Variable(torch.cuda.FloatTensor([label] * self.batch_size)).view(1, -1)

                yield batch_x, target

    def __len__(self):
        total_len = 0
        for key, value in self.data_dictionary.items():
            bound = value["bound"]
            total_len += bound
        return total_len
