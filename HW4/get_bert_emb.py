import jsonlines
import json
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from scipy import sparse

corpora = {}

with jsonlines.open('data.jsonl', 'r') as file:
    corpus = list(file)[:10000]
    for obj in corpus:
        whole_quest = obj['question'] + ' ' + obj['comment']
        answer = ''
        value = 0
        for text in obj['answers']:
            if text['author_rating']['value'] != '':
                if int(text['author_rating']['value']) > value:
                    answer = text['text']
        corpora[answer] = whole_quest

with open('mailcorpora_notlem.json', 'w', encoding='utf-8') as file:
    json.dump(corpora, file)


tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")


def get_bert_corpus(texts, model, tokenizer):
    vectors = []
    for text in texts:
        t = tokenizer(text, padding=True, truncation=True,max_length=50, add_special_tokens = True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**{k: v.to(model.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        vectors.append(embeddings[0].cpu().numpy())

    return sparse.csr_matrix(vectors)


bert_embedings = get_bert_corpus(list(corpora.values()), model, tokenizer)
sparse.save_npz('bert_matrix.npz', bert_embedings)