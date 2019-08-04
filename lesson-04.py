#coding:utf-8
import jieba
import re
from gensim.test.utils import common_texts,get_tmpfile
from gensim.models import Word2Vec,word2vec
import logging
import gensim
import pandas as pd
from langconv import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from wordcloud import WordCloud
from gensim import models
from gensim.models.word2vec import LineSentence

# def read_chunks(filepath, chunk_size=1024*1024*10):
#     file_object = open(filepath, encoding='utf-8')
#     while True:
#         chunk_data = file_object.read(chunk_size)
#         if not chunk_data:
#             break
#         yield chunk_data

#对提取的wiki语料库进行中文筛选、繁体字转换、切词
# def match(info):
#     match_words = re.findall(r'[\u4e00-\u9fa5]+', info)
#     return match_words
#
# with open('F:/wiki/AA/wiki_00.txt', 'r', encoding='utf-8') as f:
#     re_f0 = match(f.read())
#     all_words = ''.join(re_f0)
#     cut_words = jieba.cut(all_words)
#
# with open('cut_wiki_words_2019.8.4.txt', 'a', encoding='utf-8') as output:
#     for text in cut_words:
#         text_convert = Converter('zh-hans').convert(text)
#         output.write(text_convert + ' ')

#word2vec模型训练
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# sentence = word2vec.LineSentence('cut_wiki_words_2019.8.4.txt')
# model = word2vec.Word2Vec(sentence, size=100, min_count=2)
# model.save('wiki_corpus_8.4.model')
# model.wv.save_word2vec_format('wiki_corpus_2019.8.3.model')

#同义词、近义词查找
model = gensim.models.Word2Vec.load('wiki_corpus_8.4.model')
# print(model.similarity('因此', '而且'))
# print(model.similarity('漂亮', '美丽'))
# print(model.similarity('人们', '大家'))
# print(pd.Series(model.most_similar(u'漂亮')))
# print(pd.Series(model.most_similar(u'华丽')))
# print(model.wv['中国'])

# font_path = 'C:/Windows/Fonts/simsunb.ttf'
#
# def get_mask():
#     x, y = np.ogrid[:300, :300]
#     mask = (x-150)**2 + (y-150)
#     mask = 255 * mask.astype(int)
#     return mask
#
# def draw_word_cloud(word_cloud):
#     wc = WordCloud(font_path=font_path, background_color='white', mask=get_mask())
#     wc.generate_from_frequencies(word_cloud)
#     plt.axis("off")
#     plt.imshow(wc, interpolation='bilinear')
#     plt.show()
#
# def test():
#     logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#     model = models.Word2Vec.load('wiki_corpus_8.4.model')
#     one_corpus = ['因此']
#     result = model.most_similar(one_corpus[0], topn=10)
#     word_cloud = dict()
#     for sim in result:
#         word_cloud[sim[0]] = sim[1]
#     draw_word_cloud(word_cloud)
#
# if __name__ == '__main__':
#     test()

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# wiki_news = open('cut_wiki_words_2019.8.4.txt', 'r', encoding='utf-8')
# model = Word2Vec(LineSentence(wiki_news), sg=0, size=50, window=5, min_count=5)
# print('训练结束')

#词云显示
def tsne_plot(mdel):
    labels = []
    tokens = []
    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500)
    new_value = tsne_model.fit_transform(tokens)
    x = []
    y = []
    for value in new_value:
        x.append(value[0])
        y.append(value[1])
    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()

tsne_plot(model)

