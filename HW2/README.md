# ДЗ2

### Структура работы


В этот раз весь код находится в файле *app.py*
Корпус, который остался у меня с прошлой дз сохранён в файл *friends_lemmas_col.json* 

К файлу app.py поступает запрос формата -file -search, где файл - это путь к корпусу, а search - поисковой запрос. Далее функция get_tfidf() индексирует корпус, а функция index_search() очищает и индексирует сам запрос. Я решила очистить запрос от стоп-слоп и лематизировать его, потому что в корпусе все слова лематизированы и всё равно нет стоп-слов. И, соотвественно, get_cosinus() находит косинусное расстояние между запросом и каждым документом корпуса и вовзвращает упорядоченный список серий (от самой подходящей до самой неподходящей).

### Пример запроса и результата

*Запрос*: росс и рейчел были на перерыве

*Топ выдачи*: The 1 closest result is Friends - 3x15 - The One Where Ross And Rachel Take A Break (1).ru.txt
The 2 closest result is Friends - 3x16 - The One With The Morning After (2).ru.txt
The 3 closest result is Friends - 6x02 - The One Where Ross Hugs Rachel.ru.txt

_______________________________

*Запрос*: я открываю банку, а там палец

*Топ выдачи*: The 1 closest result is Friends - 1x03 - The One With The Thumb.ru.txt
The 2 closest result is Friends - 3x03 - The One With The Jam.ru.txt
The 3 closest result is Friends - 5x08 - The One With The Thanksgiving Flashbacks.ru.txt
