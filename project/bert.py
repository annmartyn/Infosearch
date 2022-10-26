import json, torch
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def index_query(query): #векторизация запроса (без обработки)
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    if type(query) == list:
        query_str = " ".join(["".join(word) for word in query])
    else:
        query_str = query
    encoded_input = tokenizer(query_str, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    return sparse.csr_matrix(embedding)


def bert_result_search(corpus, embeddings, query): #расчет сходства, поиск
    scores = cosine_similarity(embeddings, query)
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    top15 = np.array(list(corpus))[sorted_scores_indx.ravel()[:15]]
    return top15




def bert_search(some_query):
    q = index_query(some_query)
    with open("mailcorpora_notlem.json", 'r', encoding = "utf-8") as a:
        corpus_data = json.load(a)
    matrix = sparse.load_npz('bert_matrix.npz')
    return bert_result_search(corpus_data, matrix, q)
