from pymorphy2 import MorphAnalyzer
m = MorphAnalyzer()
from string import punctuation
punctuation += '—…–'
import nltk
from nltk.corpus import stopwords
stops = set(stopwords.words("russian"))

def read_lemmatize(filename):
    lemmas = []
    with open(filename, 'r', encoding='utf-8') as file_read:
        text = file_read.read().replace('\n', ' ')
        for p in punctuation:
            text = text.replace(p, '')
        text = text.lower()
        for word in text.split():
            lemmas.append(m.parse(word)[0].normal_form)
    filtered_lemmas = [word for word in lemmas if word not in stopwords.words('russian')]
    return filtered_lemmas


def get_lemmatized_dict(files):
    all_lemmatized_texts = {}
    for f in files:
        all_lemmatized_texts[f.split('/')[-1]] = read_lemmatize(f)
    return all_lemmatized_texts
