import pandas as pd
import numpy as np
import jieba
import jieba.posseg
import jieba.analyse
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
from functools import reduce
import re
import sys, codecs
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

#提取数据集词向量
word_embeddings = {}
f = open('sgns.financial.bigram-char', encoding='utf-8', errors='ignore')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

#获取每个句子的特征向量,每个向量大小为300
def get_sentence_vector(clean_sentences):
    sentence_vectord = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((300,))) for w in i.split()]) / (len(i.split())+0.001)
        else:
            v = np.zeros((300,))
        sentence_vectord.append(v)
    return sentence_vectord

#创建停用词词表
stopwords = [w.strip() for w in open('stopwords.txt', encoding='utf-8').readlines()]
#在字典中存储了467163个不同术语的词向量
# print(len(word_embeddings))

#对句子中文分词并去除停用词
def cut_sentence(sentence):
    outstr = ''
    cut_word = jieba.cut(sentence.strip())
    for word in cut_word:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += ' '
    return outstr

# 分句
def split_sentence(content):
    sentences = [sentence for sentence in re.split(r'[？?！!.。;；……\n\r]', content) if sentence]
    return sentences
#文本预处理
def clean_content(content):
    clean_sentences = []
    sens = split_sentence(content)
    for sen in sens:
        cut_sen = cut_sentence(sen)
        clean_sentences.append(cut_sen)
    return clean_sentences

#根据排名前N的句子，提取摘要
def get_key_sentence(content):
    sentences = split_sentence(content)
    clean_sentences = clean_content(content)
    sentence_vectors = get_sentence_vector(clean_sentences)
    # 创建一个n*n的空矩阵
    sim_mat = np.zeros([len(clean_sentences), len(clean_sentences)])
    # 余弦相似度初始化矩阵
    for i in range(len(clean_sentences)):
        for j in range(len(clean_sentences)):
            if i != j:
                sim_mat[i][j] = \
                cosine_similarity(sentence_vectors[i].reshape(1, 300), sentence_vectors[j].reshape(1, 300))[0, 0]

    # 将相似性矩阵sim_mat转换为图结构，图的节点为句子，边权重为句子间的相似度,得到句子排名
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(clean_sentences)), reverse=True)

    key_clean_sentences = ''
    for i in range(len(ranked_sentences)):
        if ranked_sentences[i][0] > 0.04:
            key_clean_sentences += ranked_sentences[i][1].replace(' ', '')
            key_clean_sentences += '，'

    return key_clean_sentences

print(get_key_sentence('另据美联社报道，美国暂停对定于周二生效的2500亿美元中国进口商品的关税上调，中国同意从美国购买400亿至500亿美元的农产品。美国总统特朗普表示，我们已经与中国达成了实质性的第一阶段协议，将需要五周的时间来起草协议。中国副总理刘鹤表示，我们正在朝着中美经济关系的积极方向取得很多进展。我们在许多领域都取得了实质性进展，将继续努力。不过，特朗普尚未改变12月15日对中国产品加征1600亿美元关税的计划，此举将把制裁扩大到几乎所有中国运往美国的船只。'))