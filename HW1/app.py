import get_statistics
from indexed import create_matrix
from get_files import open_and_lemmatize


def main():
    all_lemmatized_texts = open_and_lemmatize()
    X, final_matrix = create_matrix(all_lemmatized_texts)
    print('a) Самое частое слово:', get_statistics.often_word(final_matrix))
    print('b) Список самых редких слов:', get_statistics.rarest_words(final_matrix))
    print('c) Список общих слов:', get_statistics.common_words(X, all_lemmatized_texts))
    print('d) Самое популярное имя и количество раз, которое оно встречалось:', get_statistics.common_name(final_matrix))


if __name__ == '__main__':
    main()
