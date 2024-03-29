import pandas as pd
import re
import jieba
from collections import Counter
from functools import reduce
from operator import add, mul
import numpy as np

filename = 'F:/fintech/train.txt'
content = open(filename,'r',encoding='UTF-8')
articles = content.read()

def token(string):
    return re.findall('[\u4e00-\u9fa5]+', string)

articles_ = [''.join(token(str(a))) for a in articles]
articles_clean = [i for i in articles_ if i!='']
list_article = ''.join(articles_clean)
print(list_article)

def cut(string):
    return list(jieba.cut(string))

TOKEN = cut(list_article)
print(TOKEN)
words_count = Counter(TOKEN)
print(words_count.most_common(10))

TOKEN = [str(t) for t in TOKEN]
TOKEN_2_GRAM = [''.join(TOKEN[i:i+2]) for i in range(len(TOKEN[:-2]))]
words_count_2 = Counter(TOKEN_2_GRAM)

def prob_1(word): return words_count[word] / len(TOKEN)

def prob_2(word1, word2):
    if word1 + word2 in words_count_2: return words_count_2[word1+word2] / len(TOKEN_2_GRAM)
    else:
        return 1 / len(TOKEN_2_GRAM)

def get_probablity(sentence):
    words = cut(sentence)
    sentence_pro = 1
    for i, word in enumerate(words[:-1]):
        next_ = words[i + 1]
        probability = prob_2(word, next_)
        sentence_pro *= probability
    return sentence_pro

print(get_probablity('我想买保险'))
print(get_probablity('我想买股票'))

need_compared = [
    "今天晚上请你吃大餐，我们一起吃日料 明天晚上请你吃大餐，我们一起吃苹果",
    "真事一只好看的小猫 真是一只好看的小猫",
    "今晚我去吃火锅 今晚火锅去吃我",
    "洋葱奶昔来一杯 养乐多绿来一杯"
]

for s in need_compared:
    s1, s2 = s.split()
    p1, p2 = get_probablity(s1), get_probablity(s2)

    better = s1 if p1 > p2 else s2

    print('{} is more possible'.format(better))
    print('-' * 4 + ' {} with probility {}'.format(s1, p1))
    print('-' * 4 + ' {} with probility {}'.format(s2, p2))