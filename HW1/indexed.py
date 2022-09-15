from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def create_matrix(texts):
    vectorizer = CountVectorizer(analyzer='word')
    texts_corpora = [' '.join(x) for x in texts.values()]
    X = vectorizer.fit_transform(texts_corpora)
    matrix_freq = np.asarray(X.sum(axis=0)).ravel()
    final_matrix = np.array([np.array(vectorizer.get_feature_names()), matrix_freq])
    return X, final_matrix

