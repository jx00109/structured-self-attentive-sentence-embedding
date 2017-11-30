# -*- encoding:utf8 -*-
from utils import Dictionary
from model import Classifier
import torch
import torch.nn as nn
import torch.optim as optim
import utils


# ****************************** 参数设置 ******************************************* #
DICT_PATH = './preprocessed_data/mydict(all).json'
DATA_PATH = './preprocessed_data/data(all).json'
MODEL_PATH = './models/bestmodel-%f-%f.pt'
LOG_PATH = './logs/log.txt'
BATCH_SIZE = 50
DROPOUT = 0.3
N_LAYERS = 2
HIDDEN_DIM = 300
EMBED_DIM = 50
POOLING = 'all'
ATTENTION_UNIT = 350
ATTENTION_HOPS = 4
MAX_LEN = 500
PRETRAINED_WORDVEC = True
N_HIDDEN_UNITS = 300
N_CLASSES = 5
LR = 0.001
EPOCH = 10
USE_ATTENTION = True
PENALIZATION_COEFF = 1.0
CLIP = 0.5
N_TRAIN = 50000
N_DEV = 2000
N_VAL = 2000
EARLY_STOP = 5
# ********************************************************************************** #
# 计算矩阵的F范数
def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), 2), 1).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')


def evaluate(model, loss_func, dictionary, data):
    """evaluate the model while training"""
    model.eval()  # turn on the eval() switch to disable dropout
    total_loss = 0
    total_correct = 0
    for texts, labels, masks, bsz in utils.getBatch(data=data, dictionary=dictionary, maxlen=MAX_LEN,
                                                    batch_size=BATCH_SIZE):
        hidden = model.init_hidden(texts.size(0))
        output, attention = model.forward(texts, masks, hidden)
        output_flat = output.view(texts.size(0), -1)
        total_loss += loss_func(output_flat, labels).data
        prediction = torch.max(output_flat, 1)[1]
        total_correct += torch.sum((prediction == labels).float())
    return total_loss[0] / (len(data) // BATCH_SIZE), total_correct.data[0] / len(data)


def train(model, loss_func, dictionary, epoch, train_data, dev_data, identity_mat, stop_counter):
    global best_dev_loss, best_acc
    model.train()
    total_loss = 0
    for texts, labels, masks, bsz in utils.getBatch(data=train_data, dictionary=dictionary, maxlen=MAX_LEN,
                                                    batch_size=BATCH_SIZE):
        init_state = model.init_hidden(bsz)
        pred, attention = model.forward(sents=texts, mask=masks, init_hc=init_state)

        loss = loss_func(pred.view(texts.size(0), -1), labels)

        if USE_ATTENTION:
            attentionT = torch.transpose(attention, 1, 2).contiguous()
            extra_loss = Frobenius(torch.bmm(attention, attentionT) - identity_mat[:attention.size(0)])
            loss += PENALIZATION_COEFF * extra_loss

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm(model.parameters(), CLIP)
        optimizer.step()

        total_loss += loss.data

    dev_loss, acc = evaluate(model, loss_func, dictionary, dev_data)
    res = 'epoch: %d, dev loss: %f, acc: %f' % (epoch + 1, dev_loss, acc)
    print(res)
    utils.saveLog(LOG_PATH, res)
    utils.div('-')

    if not best_acc or acc > best_acc:
        with open(MODEL_PATH % (dev_loss, acc), 'wb') as f:
            torch.save(model, f)
        best_acc = acc
        stop_counter = 0
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.2
        if EARLY_STOP != 0:
            stop_counter += 1

    return stop_counter


dictionary = Dictionary(path=DICT_PATH)

n_token = len(dictionary)

best_dev_loss = None
best_acc = None

model = Classifier({
    'dropout': DROPOUT,
    'n_tokens': n_token,
    'n_layers': N_LAYERS,
    'hidden_dim': HIDDEN_DIM,
    'embed_dim': EMBED_DIM,
    'pooling': POOLING,
    'dictionary': dictionary,
    'pretrained_wordvec': PRETRAINED_WORDVEC,
    'attention-unit': ATTENTION_UNIT,
    'attention-hops': ATTENTION_HOPS,
    'nfc': N_HIDDEN_UNITS,
    'n_classes': N_CLASSES
})

# 得到单位矩阵
I = utils.getIdentityMatrix(BATCH_SIZE, ATTENTION_HOPS)

# 损失函数 交叉熵
loss_func = nn.CrossEntropyLoss()

# 优化算法
optimizer = optim.Adam(model.parameters(), lr=LR, betas=[0.9, 0.99], eps=1e-8, weight_decay=0)

print('Begin to load data...')
data_train, data_dev, data_val = utils.divideData(DATA_PATH, N_TRAIN, N_DEV, N_VAL)

print('number of train data: %d' % len(data_train))
print('number of develop data: %d' % len(data_dev))
print('number of valid dataL %d' % len(data_val))

counter = 0
for epoch in range(EPOCH):
    counter = train(model, loss_func, dictionary, epoch, data_train, data_dev, I, counter)
    if counter == EARLY_STOP:
        break

test_loss, acc = evaluate(model, loss_func, dictionary, data_val)
res = 'testing model, dev loss: %f, acc: %f' % (test_loss, acc)
utils.saveLog(LOG_PATH, res)
print(res)
