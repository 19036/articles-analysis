import sqlite3
import sys, os, time
import json  # для конвертации словарь <--> строка: json.dumps(dict), json.loads(str)
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile

db_path = '../../main_db/articles.db'
logs_path = ('../logs/doc2vec/' + time.ctime().replace(' ', '___') + '_doc2vec.txt').replace(':', '-')

try:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
except:
    print(f"Couldn't connect to database: {db_path}")


def invert_abstract(inv_index):
    " Выдаёт строку -- абстракт по abstract_inverted_index "
    if inv_index is not None:
        l_inv = [(w, p) for w, pos in inv_index.items() for p in pos]
        return " ".join(map(lambda x: x[0], sorted(l_inv, key=lambda x: x[1])))


def write_logs(message:str, time_data=True):
    
    global logs_path
    s = ''
    if time_data:
        s = f'[{time.ctime()}]   '
    s += message + '\n'
    with open(logs_path, 'a') as f:
        f.write(s)
    print(s)


def get_texts():
    
    path = 'sample_texts.txt'
    
    t_start = time.time()
    cursor.execute('''
                   SELECT cleaned_abstract
                   FROM articles
                   WHERE level = ?
                   AND cleaned_abstract IS NOT NULL
                   ''', (1,))
    
    results = cursor.fetchall()
    
    texts = []
    for res in results:
        text = invert_abstract(json.loads(res[0])).split(' ')
        texts.append(text)
    
    texts = json.dumps(texts)
    with open(path, 'w') as f:
        f.write(texts)
    
    t = round(time.time() - t_start, 1)
    print(f'Got {len(results)} texts in {t} sec')
    

def get_all_texts():
    
    """ Не запускать)) """
    
    path = 'all_texts.txt'
    
    t_start = time.time()
    cursor.execute('''
                   SELECT id, cleaned_abstract
                   FROM articles
                   WHERE cleaned_abstract IS NOT NULL
                   ''')
    
    results = cursor.fetchall()
    
    texts = []
    for id_, abstract in results:
        abstract = invert_abstract(json.loads(abstract)).split(' ')
        texts.append((id_, abstract))
    
    texts = json.dumps(texts)
    with open(path, 'w') as f:
        f.write(texts)
    t = round(time.time() - t_start, 1)
    print(f'Got {len(results)} texts in {t} sec')
    
    



def train_doc2vec():
    
    path = 'sample_texts.txt'
    
    step = 100
    t_start = time.time()
    
    with open(path, 'r') as f:
        texts = f.read()
    
    texts = json.loads(texts)[:1000]
    
    n = len(texts)
    train_data = texts[:step]
    count = 1
    documents = [TaggedDocument(doc, [str(i)]) for i, doc in enumerate(train_data)]
    model = Doc2Vec(documents, vector_size=100, window=5, min_count=5, workers=2, epochs=50)
    model.save('doc2vec.model')
    t = round(time.time() - t_start, 1)
    print(f'count package: {t} sec')
    t_start = time.time()
    while len(texts) > 0:
        model = Doc2Vec.load('doc2vec.model')
        texts = texts[step:]
        train_data = texts[:step]
        documents = [TaggedDocument(doc, [str(i+step*count)]) for i, doc in enumerate(train_data)]
        model.train(documents, total_examples=len(train_data), epochs=model.epochs)
        count += 1
        t = round(time.time() - t_start, 1)
        print(f'count package: {t} sec')
        t_start = time.time()
        model.save('doc2vec.model')
    
    



if __name__ == '__main__':
    
    # get_texts()
    # train_doc2vec()
    
    pass







