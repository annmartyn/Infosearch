import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def often_word(final_matrix): #получаем самое частое слово
    quant = [int(a) for a in final_matrix[1]]
    return final_matrix[0][quant.index(max(quant))]


def rarest_words(final_matrix): #список редких слов
    quant = [int(a) for a in final_matrix[1]]
    rare_index = [num for num, q in enumerate(quant) if q == 1]
    list_of_rares = [final_matrix[0][r] for r in rare_index]
    return list_of_rares


def common_words(X, texts):
    new_x = np.rot90(X.toarray())
    vectorizer = CountVectorizer(analyzer='word')
    texts_corpora = [' '.join(x) for x in texts.values()]
    X = vectorizer.fit_transform(texts_corpora)
    common_words_list = []
    for num, r in enumerate(new_x):
        if 0 not in list(r):
            common_words_list.append(vectorizer.get_feature_names()[-num - 1])
    return common_words_list


def get_name_count(name_vars, final_matrix):
    name_indexes = [list(final_matrix[0]).index(a) for a in name_vars]
    name_counter = 0
    for n in name_indexes:
        name_counter += int(list(final_matrix[1])[n])
    return name_counter


def common_name(final_matrix):
    name_counters = dict()
    name_counters['Моника'] = get_name_count(['моника', 'мона'], final_matrix)
    name_counters['Чендлер'] = get_name_count(['чендлер', 'чен', 'чэндлер'], final_matrix)
    name_counters['Росс'] = get_name_count(['росс'], final_matrix)
    name_counters['Рэйчел'] = get_name_count(['рэйчел', 'рейч'], final_matrix)
    name_counters['Джоуи'] = get_name_count(['джой', 'джо'], final_matrix)
    name_counters['Фиби'] = get_name_count(['фиби', 'фибс'], final_matrix)
    for k, v in name_counters.items():
        if v == max(name_counters.values()):
            answer = [k, v]
            return answer