# -*- encoding:utf8 -*-
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchtext.vocab as vocab
import numpy as np
import utils as u


# 定义双向LSTM模型
class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.dropout = nn.Dropout(config['dropout'])  # 一定概率将输入的部分维度值置0
        self.encoder = nn.Embedding(config['n_tokens'], config['embed_dim'])  # 构建嵌入矩阵 [n_tokens x embed_dim]
        self.n_layers = config['n_layers']  # RNN的层数
        self.hidden_dim = config['hidden_dim']  # 隐藏层维度

        self.pooling = config['pooling']

        self.dictionary = config['dictionary']
        self.init_weights()  # 初始化词向量矩阵
        self.encoder.weight.data[self.dictionary.word2idx['<pad>']] = 0  # 将<pad>对应的向量设置为全0
        self.bilstm = nn.LSTM(
            input_size=config['embed_dim'],
            hidden_size=config['hidden_dim'],
            num_layers=config['n_layers'],
            dropout=config['dropout'],
            bidirectional=True,
            batch_first=True
        )

        if config['pretrained_wordvec']:
            self.load_pretrained_vectors(config['embed_dim'])

    def forward(self, sents, mask, init_hc):
        emb = self.dropout(self.encoder(sents))
        h = self.bilstm(emb, init_hc)[0]  # 前向传播 [fh0, fh1, ...,fhT]

        h = torch.mul(h, mask[:, :, None])  # [batch size, seq len, hidden_dim]

        # if self.pooling == 'mean':
        #     outh = torch.mean(outh, 0).squeeze()
        # elif self.pooling == 'max':
        #     outh = torch.max(outh, 0)[0].squeeze()
        # elif self.pooling == 'all' or self.pooling == 'all-word':
        #     outh = torch.transpose(outh, 0, 1).contiguous()

        return h

    # 初始化词向量
    def init_weights(self, init_range=0.1):
        self.encoder.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_()),
                Variable(weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_()))

    # 载入预训练词向量
    def load_pretrained_vectors(self, embed_dim):
        glove = vocab.GloVe(name='6B', dim=50)  # 使用torchtext下载glove
        print('loaded %d words' % len(glove.stoi))
        assert len(glove.vectors[0]) == embed_dim, '预训练词向量维度与超参数不符！'

        n_loaded = 0  # 记录载入的词数
        for word in self.dictionary.word2idx:
            # 如果单词不在glove的词典中，跳过
            if word not in glove.stoi:
                continue
            real_id = self.dictionary.word2idx[word]  # 单词在嵌入矩阵中的真实索引
            loaded_id = glove.stoi[word]  # 单词在glove词向量矩阵中的索引值
            self.encoder.weight.data[real_id] = glove.vectors[loaded_id]  # 将单词对应行的向量替换成glove向量
            n_loaded += 1
        print('%d words have been loaded.' % n_loaded)


class SelfAttentiveEncoder(nn.Module):
    def __init__(self, config):
        super(SelfAttentiveEncoder, self).__init__()
        self.bilstm = BiLSTM(config)
        self.dropout = nn.Dropout(config['dropout'])
        self.ws1 = nn.Linear(config['hidden_dim'] * 2, config['attention-unit'], bias=False)
        self.ws2 = nn.Linear(config['attention-unit'], config['attention-hops'], bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.dictionary = config['dictionary']
        self.init_weights()
        self.attention_hops = config['attention-hops']

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, sents, mask, init_hc):
        outh = self.bilstm.forward(sents=sents, mask=mask, init_hc=init_hc)  # [bsz, seq_len, hidden_dim]
        size = outh.size()
        compressed_embed = outh.view(-1, outh.size()[2])  # [bsz*seq_len, hidden_dim]

        hbar = self.tanh(self.ws1(self.dropout(compressed_embed)))  # [bsz*seq_len, attention-units]

        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, seq_len, hops]

        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hops, seq_len]

        alphas = self.softmax(alphas.view(-1, size[1]))  # [bsz*hop, seq_len]

        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, seq_len]

        return torch.bmm(alphas, outh), alphas

    def init_hidden(self, bsz):
        return self.bilstm.init_hidden(bsz)


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.encoder = SelfAttentiveEncoder(config)
        self.fc = nn.Linear(config['hidden_dim'] * 2 * config['attention-hops'], config['nfc'])

        self.dropout = nn.Dropout(config['dropout'])
        self.tanh = nn.Tanh()
        self.pred = nn.Linear(config['nfc'], config['n_classes'])
        self.dictionary = config['dictionary']
        self.init_weights()

    def init_weights(self, init_range=0.1):
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.fill_(0)
        self.pred.weight.data.uniform_(-init_range, init_range)
        self.pred.bias.data.fill_(0)

    def forward(self, sents, mask, init_hc):
        outh, attention = self.encoder.forward(sents, mask, init_hc)
        outh = outh.view(outh.size(0), -1)

        fc = self.tanh(self.fc(self.dropout(outh)))
        pred = self.pred(self.dropout(fc))

        return pred, attention

    def init_hidden(self, bsz):
        return self.encoder.init_hidden(bsz)