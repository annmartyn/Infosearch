import jsonlines
import json
from pymorphy2 import MorphAnalyzer
m = MorphAnalyzer()
corpora = {}


def lemmatize_search(text):
    lemmas = []
    for p in punctuation:
        text = text.replace(p, '')
    text = text.lower()
    for word in text.split():
        lemmas.append(m.parse(word)[0].normal_form)
    filtered_lemmas = [word for word in lemmas if word not in stopwords.words('russian')]
    return ' '.join(filtered_lemmas)


with jsonlines.open('data.jsonl', 'r') as file:
    corpus = list(file)[:50000]
    for obj in tqdm(corpus):
        whole_quest = obj['question'] + ' ' + obj['comment']
        answer = ''
        value = 0
        for text in obj['answers']:
            if text['author_rating']['value'] != '':
                if int(text['author_rating']['value']) > value:
                    answer = text['text']
        corpora[answer] = lemmatize_search(whole_quest)


with open('mailcorpora.json', 'w', encoding='utf-8') as file:
    json.dump(corpora, file)