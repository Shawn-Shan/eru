from __future__ import absolute_import

import unicodedata
import re
import os
from torch.autograd import Variable


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"(')", r"", s)
    s = re.sub(r"[0-9]+", "NUM", s)
    s = re.sub(r"([.!?,:-;\"])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?,-:;\"]+", r" ", s)
    return s


def fix_review(review):
    sent = normalize_string(review)
    sep = re.findall(r'([a-z][^\.!?]*[\.!? ]*)', sent)
    sep = ["SOS " + s + " EOS" for s in sep]
    return "SOP " + " ".join(sep) + " EOP"


def init_gpu(gpu_num):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data


def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)
