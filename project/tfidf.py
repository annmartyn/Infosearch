from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re, pymorphy2, json
import numpy as np
morph = pymorphy2.MorphAnalyzer()


def clear_and_morhp(some_text): #предобработка текста
    cl_text = some_text.lower().replace('\n', ' ')
    match = re.compile(r'[^\w\s]')
    cl_text_1 = match.sub(' ', cl_text)
    lemmas = [morph.parse(token)[0].normal_form for token in cl_text_1.split()]
    return " ".join(lemmas)


def index_query(query, vctrzr):
    if type(query) == list:
        query_str = " ".join(["".join(word) for word in query])
    else:
        query_str = query
    qarray = vctrzr.transform([clear_and_morhp(query_str)]).toarray()
    return np.squeeze(np.asarray(qarray))


def fit_vectorizer(texts):
    corpus_text = ["".join(lems) for lems in texts.values()]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform(corpus_text)
    return tfidf_vectorizer


def reverse_index_mtrx(init_vcb): #функция индексации корпуса, на выходе которой посчитанная матрица Document-Term
    vectorizer = TfidfVectorizer()
    corpus_text = ["".join(lems) for lems in init_vcb.values()]
    X = vectorizer.fit_transform(corpus_text)
    return X


def tfidf_result_search(json_data, query_vec_):
    corpus_mtrx_ = reverse_index_mtrx(json_data)
    scores = cosine_similarity(corpus_mtrx_, [query_vec_])
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    top15 = np.array(list(json_data.keys()))[sorted_scores_indx.ravel()[:15]]
    return top15


def tfidf_search(some_query):
    with open("mailcorpora_lem.json", 'r', encoding = "utf-8") as a:
        pr_corpus_data = json.load(a)
    q3 = index_query(some_query, fit_vectorizer(pr_corpus_data))
    query_answers = tfidf_result_search(pr_corpus_data, q3)
    return query_answers

