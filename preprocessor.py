import json
import string
from utils import Dictionary

from nltk.tokenize import word_tokenize

# ********************************************************************* #
datapath = './data/dataset/review.json'
outpath = './preprocessed_data/data(%s).json'
dictpath = './preprocessed_data/mydict(%s).json'
debug_flag = True

# ********************************************************************* #

mydict = Dictionary()
mydict.add_word('<pad>')

reviews = open(datapath).readlines()
n_reviews = len(reviews)
print('%d条评论将被载入...' % n_reviews)

if debug_flag:
    size = '5'
else:
    size = 'all'

with open(outpath % size, 'a') as f:
    for i, line in enumerate(reviews):
        if debug_flag:
            if i == 5:
                break
        json_data = json.loads(line)
        words = word_tokenize(json_data['text'].lower())
        only_words = list()
        for word in words:
            # 去除标点和数字
            if word in string.punctuation or word.isdigit():
                continue
            else:
                only_words.append(word)

        data = {
            'label': json_data['stars'],
            'text': only_words
        }

        f.write(json.dumps(data) + '\n')

        for word in only_words:
            mydict.add_word(word)
        if i % 100 == 99:
            print('%.2f%% done, dictionary size: %d' % ((i + 1) * 100 / n_reviews, len(mydict)))

# 保存字典，下次可以直接载入
with open(dictpath % size, 'a') as f:
    f.write(json.dumps(mydict.idx2word) + '\n')
    f.close()
