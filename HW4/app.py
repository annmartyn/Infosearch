import argparse
from transformers import AutoTokenizer, AutoModel
import numpy as np
import json
import torch
from scipy import sparse


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

def search(corpus, embeddings, query):
    scores = np.dot(embeddings, query.T).toarray()
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    corpus = np.array(corpus)[sorted_scores_indx.ravel()]
    return list(corpus)[:10]


def main():
    with open('../mailcorpora_notlem.json', 'r', encoding='utf-8') as file:
        corpora = json.loads(file.read())
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('search', nargs='+', type=str)
    args = parser.parse_args()
    bert_embedings = sparse.load_npz(args.file)
    bert_query = get_bert_corpus([args.search], model, tokenizer)
    result = search(list(corpora.keys()), bert_embedings, bert_query)
    for num, el in enumerate(result):
        print(f'{num}. {el}')



if __name__ == '__main__':
    main()