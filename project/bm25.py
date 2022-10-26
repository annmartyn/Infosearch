from pymorphy2 import MorphAnalyzer
m = MorphAnalyzer()
import json

from scipy import sparse
import operator, functools
from string import punctuation
punctuation += '—…–'
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
stops = set(stopwords.words("russian"))
import numpy as np


#здесь я открываю уже обработанный корпус, чтобы не обрабатывать 50 000 текстов, обработка лежит в файл
# collect_corpora.py

count_vectorizer = CountVectorizer()
tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')


def lemmatize_search(text):
    lemmas = []
    for p in punctuation:
        text = text.replace(p, '')
    text = text.lower()
    for word in text.split():
        lemmas.append(m.parse(word)[0].normal_form)
    filtered_lemmas = [word for word in lemmas if word not in stopwords.words('russian')]
    return ' '.join(filtered_lemmas)


def get_indexed_corpora(texts):  # индексация корпуса
    k = 2
    b = 0.75
    x_count_vec = count_vectorizer.fit_transform(texts)
    x_tf_vec = tf_vectorizer.fit_transform(texts)
    tfidf_vectorizer.fit_transform(texts)
    idf = tfidf_vectorizer.idf_
    idf = np.expand_dims(idf, axis=0)
    tf = x_tf_vec
    len_d = x_count_vec.sum(axis=1)
    avdl = len_d.mean()
    B_1 = (k * (1 - b + b * len_d / avdl))
    B_1 = np.expand_dims(B_1, axis=-1)
    rows, cols, values = list(), list(), list()
    for i, j in zip(*tf.nonzero()):
        rows.append(i)
        cols.append(j)
        A = functools.reduce(operator.mul, [idf[0][j], tf[i, j], (k + 1)])
        B = tf[i, j] + B_1[i]
        values.append((A / B)[0][0])
    return sparse.csr_matrix((values, (rows, cols)))


def get_indexed_search(query):  # обработка запроса
    return tfidf_vectorizer.transform([lemmatize_search(query)])


def get_answers(corpus, query_vec, corpus_matrix):  # функция поиска ответов
    top = []
    scores = np.dot(corpus_matrix, query_vec.T).toarray()
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    for i in sorted_scores_indx.ravel()[:10]:
        top.append(list(corpus)[i])
    return top


def bm25_search(query):
    with open('mailcorpora_lem.json', 'r', encoding='utf-8') as file:
        corpora = json.load(file)
    corp = np.array(list(corpora.values()))
    corpus_doc_names = np.array(list(corpora.keys()))
    corpus_matrix = get_indexed_corpora(corp)
    query = get_indexed_search(query)
    return get_answers(corpus_doc_names, query, corpus_matrix)
