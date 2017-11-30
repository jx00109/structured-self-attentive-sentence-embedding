import json
import numpy as np
from torch.autograd import Variable
import torch
import random


class Dictionary(object):
    def __init__(self, path=''):
        self.word2idx = dict()
        self.idx2word = list()
        if path != '':  # load an external dictionary
            words = json.loads(open(path, 'r').readline())
            for item in words:
                self.add_word(item)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def div(s):
    print(s * 90)


def getIdentityMatrix(batch_size, attention_hops):
    I = Variable(torch.zeros(batch_size, attention_hops, attention_hops))
    for i in range(batch_size):
        for j in range(attention_hops):
            I.data[i][j][j] = 1
    return I


def getBatch(data, dictionary, maxlen, batch_size):
    raw_data = [json.loads(s) for s in data]

    raw_data_in_ids = [[dictionary.word2idx[xx] for xx in x['text']] for x in raw_data]
    raw_targets = [x['label'] - 1 for x in raw_data]

    # 统计该批次最长序列的长度
    mlen = 0
    for line in raw_data_in_ids:
        mlen = max(mlen, len(line))

    mlen = min(mlen, maxlen)

    mask = np.zeros(shape=[len(data), mlen])

    for i in range(len(raw_data)):
        if len(raw_data_in_ids[i]) > mlen:
            raw_data_in_ids[i] = raw_data_in_ids[i][:mlen]

            mask[i][:mlen] = 1
        else:
            mask[i][:len(raw_data_in_ids[i])] = 1
            for _ in range(mlen - len(raw_data_in_ids[i])):
                raw_data_in_ids[i].append(dictionary.word2idx['<pad>'])
    for batch, i in enumerate(range(0, len(raw_data), batch_size)):
        texts = Variable(torch.LongTensor(raw_data_in_ids[i:i + batch_size]))

        labels = Variable(torch.LongTensor(raw_targets[i:i + batch_size]))
        masks = Variable(torch.FloatTensor(mask[i:i + batch_size]))
        tbatch_size = len(raw_data_in_ids[i:i + batch_size])
        yield texts, labels, masks, tbatch_size


def divideData(src, train, dev, val):
    data = open(src).readlines()
    n_samples = len(data)

    sidx = np.random.permutation(n_samples)

    train_data = [data[s] for s in sidx[:train]]
    dev_data = [data[s] for s in sidx[train:train + dev]]
    val_data = [data[s] for s in sidx[train + dev:train + dev + val]]

    return train_data, dev_data, val_data


def saveLog(path, s):
    with open(path, 'a') as f:
        f.write(s + '\n')
