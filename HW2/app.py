import json
from pymorphy2 import MorphAnalyzer
m = MorphAnalyzer()
import argparse
from string import punctuation
punctuation += '—…–'
from nltk.corpus import stopwords
stops = set(stopwords.words("russian"))
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import dot
from numpy.linalg import norm


def get_tfidf(corp): #функция индексации корпуса
    with open(corp, 'r', encoding='utf-8') as c:
        corpora = json.load(c)
    corpora_vectors = {}
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([' '.join(x) for x in corpora.values()])
    for i in range(len(X.toarray())):
        corpora_vectors[list(corpora.keys())[i]] = X.toarray()[i]
    return corpora_vectors, vectorizer


def index_search(text, vectorizer): #функция индексации поиска
    text = ' '.join(text)
    lemmas = []
    for p in punctuation:
        text = text.replace(p, '')
    text = text.lower()
    for word in text.split():
        lemmas.append(m.parse(word)[0].normal_form)
    filtered_lemmas = [word for word in lemmas if word not in stopwords.words('russian')]
    filtered = [' '.join(filtered_lemmas)]
    return vectorizer.transform(filtered).toarray()


def get_cosinus(search, corpora): #функция вычислений косинусных расстояний
    distances_by_series = {}
    for serie, vect in corpora.items():
        r = dot(search, vect)/(norm(search)*norm(vect))
        distances_by_series[serie] = r[0]
        sorted_tuples = sorted(distances_by_series.items(), key=lambda item: item[1], reverse=True)
        res = [a[0] for a in sorted_tuples]
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('search', nargs='+', type=str)
    args = parser.parse_args()
    indexed_corpora, vectorizer = get_tfidf(args.file)
    indexed_str = index_search(args.search, vectorizer)
    result = get_cosinus(indexed_str, indexed_corpora)
    for num, s in enumerate(result):
        print(f'The {num+1} closest result is {s}')