from flask import Flask, request, render_template
from bert import bert_search
from bm25 import bm25_search
from tfidf import tfidf_search
import time

app = Flask(__name__)

@app.route('/')
def index():
    t = ''
    header = ''
    start_time = time.time()
    if request.args:
        query = request.args['query']
        if query == "":
            answers = ["Ничего не найдено"]
        else:
            method = request.args['veсtorizer']
            if method == "BERT":
                answers = bert_search(query)
            elif method == "bm25":
                answers = bm25_search(query)
            elif method == "tf-idf":
                answers = tfidf_search(query)
            else:
                answers = ["Ничего не найдено"]
            header = "Результаты по запросу '" + query + "' через метод " + method
        end = time.time()
        t = "Время работы программы: " + str(float('{:.3f}'.format(end - start_time))) + 's'
        return render_template('index.html', answers=answers, header=header, time=t)
    return render_template('index.html', answers=[], header=header, time=t)


if __name__ == '__main__':
    app.run(debug=True)