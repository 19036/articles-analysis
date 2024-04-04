import sqlite3
import sys, os, time
import json  # для конвертации словарь <--> строка: json.dumps(dict), json.loads(str)
import gensim
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from helpfun import write_logs_by_curr_date, abstract_by_inv_index, sql_execute
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
import joblib

# db_path = '../databases/articles.db'
db_path = '../databases/sob_articles.db'
date = '_'.join(time.ctime().split(' ')[:3])
logs_path = f'../logs/ml_models/{date}.txt'
train_data_path = '../ml_models/train_data/'
models_path = '../ml_models/'

try:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
except:
    print(f"Couldn't connect to database: {db_path}")

def write_logs(message:str):
    write_logs_by_curr_date(message, logs_path)

def compose_train_data():
    
    name = 'sample_texts'
    name = 'sob_all'
    
    path = train_data_path + name + '.txt'
    
    # request = '''
    #             SELECT id, cleaned_abstract
    #             FROM articles
    #             WHERE level = 1
    #             AND cleaned_abstract IS NOT NULL
    #             '''
    
    request = '''
                SELECT id, cleaned_abstract
                FROM articles
                WHERE cleaned_abstract IS NOT NULL
                '''
        
    response = sql_execute(request, path = db_path)
    if 'error' not in response:
        results = response['results']
        t = round(response['time'], 1)
    else:
        print(response['error'])
    
    texts = []
    for id_, text in results:
        text = abstract_by_inv_index(text)
        texts.append((id_, text))
    
    texts = json.dumps(texts)
    with open(path, 'w') as f:
        f.write(texts)
    
    print(f'Got {len(results)} texts in {t} sec')
    

def get_train_data(name:str='nsu_level_1'):
    
    # name = 'level_1'
    path = train_data_path + name + '.txt'
    with open(path, 'r') as f:
        data = f.read()
    data = json.loads(data)
    return data


def compose_frequencies(name:str='level_1'):
    
    # name = 'level_1'
    path = train_data_path + name + '_freq.txt'
    
    t_start = time.time()
    data = list(map(lambda x: x[1], get_train_data(name)))
    freq = {}
    for text in data:
        for word in text.split(' '):
            if word not in freq:
                freq[word] = 1
            else:
                freq[word] += 1
    
    freq_sorted = dict(sorted(freq.items(), key=lambda elem: elem[1], reverse=True))
    with open(path, 'w') as f:
        f.write(json.dumps(freq_sorted))
    
    t = round(time.time() - t_start, 1)
    print(f'Frequencies for train_data "{name}" composed in {t} sec')
    

def get_frequencies(name:str='level_1'):
    
    # name = 'level_1'
    path = train_data_path + name + '_freq.txt'
    
    with open(path, 'r') as f:
        freq = f.read()
    
    freq = json.loads(freq)
    
    return freq


def train_doc2vec(name:str='nsu_level_1'):
    
    # name = 'level_1'
    m_path = models_path + 'doc2vec_models/dv_' + name + '.model'
    
    t_start = time.time()
    
    train_data = get_train_data(name)
    # freq = get_frequencies(name)
    
    documents = [TaggedDocument(text.split(' '), [id_]) for id_, text in train_data]
    model = Doc2Vec(documents, vector_size=100, window=5, min_count=5, workers=2, epochs=50)
    # empty_docs = [TaggedDocument([], [id_]) for id_, text in train_data]
    # model = Doc2Vec(empty_docs, vector_size=100, window=5, min_count=5, workers=2, epochs=50)
    # model.build_vocab_from_freq(freq)
    # model.train(documents, total_examples=len(documents), epochs=model.epochs)
    model.save(m_path)
    
    t = round(time.time() - t_start, 1)
    s = f'Doc2Vec model on training_data: {name} trained (traditional way) in {t} sec'
    write_logs(s)
    


def train_doc2vec_by_blocks(name:str='level_1'):
    
    # name = 'level_1'
    path = models_path + 'doc2vec_' + name + '.model'
    
    step = 1000
    t_start = time.time()
    
    with open(path, 'r') as f:
        texts = f.read()
    
    texts = json.loads(texts)[:10000]
    # texts[-1] = ['syka']*100 + ['review']*5
    # for i in range(1001, 10000):
    #     texts[i] = ['syka']*100 + ['review']*5
    # texts[0] = ['syka']*10
    
    n = len(texts)
    train_data = texts[:step]
    count = 1
    # documents = [TaggedDocument(doc, [str(i)]) for i, doc in enumerate(train_data)]
    # empty_docs = [TaggedDocument([], [str(i)]) for i in range(step, n)]
    # documents = [TaggedDocument([], [str(i)]) for i in range(n)]
    # model = Doc2Vec.load('doc2vec.model')
    # model.build_vocab
    # documents.extend(empty_docs)
    # model = Doc2Vec(documents, vector_size=100, window=5, min_count=5, workers=2, epochs=50)
    # model = Doc2Vec(vector_size=100, window=5, min_count=5, workers=2, epochs=50)
    # with open('word_freq.txt', 'r') as f:
    #     freq = json.loads(f.read())
    # model.build_vocab(documents)
    # model.build_vocab_from_freq(freq)
    
    # model.save('doc2vec.model')
    # return
    time.sleep(0.2)
    # model = Doc2Vec.load('doc2vec.model')
    t = round(time.time() - t_start, 1)
    print(f'count package: {t} sec')
    t_start = time.time()
    while len(texts) > 0:
        model = Doc2Vec.load('doc2vec.model')
        texts = texts[step:]
        train_data = texts[:step]
        documents = [TaggedDocument(doc, [str(i+step*count)]) for i, doc in enumerate(train_data)]
        # model.build_vocab(documents)
        model.train(documents, total_examples=len(train_data), epochs=model.epochs)
        count += 1
        t = round(time.time() - t_start, 1)
        print(f'count package: {t} sec')
        t_start = time.time()
        model.save('doc2vec.model')
    
    
def load_doc2vec(name:str='sob_all'):
    
    m_path = models_path +  'doc2vec_models/dv_' + name + '.model'
    model = Doc2Vec.load(m_path)
    return model

def load_birch(name:str='sob_all', n_clusters=5):
    
    m_path = models_path +  'birch/birch_' + name + f'_n={n_clusters}.pkl'
    model = joblib.load(m_path)
    return model

def load_kmeans(name:str='sob_all', n_clusters=5):
    
    m_path = models_path +  'KMeans/kmeans_' + name + f'_n={n_clusters}.pkl'
    model = joblib.load(m_path)
    return model



def train_birch(name:str='sob_all', n_clusters=3, size=10000):
    
    d2v_model = load_doc2vec(name)
    vectors = d2v_model.dv.get_normed_vectors()[:size]
    t_start = time.time()
    model = Birch(n_clusters=n_clusters)
    model.fit(vectors)
    m_path = models_path + 'birch/birch_' + name + f'_n={n_clusters}.pkl'
    joblib.dump(model, m_path)
    print(f'Birch trained in {round(time.time()-t_start, 1)} sec')
    
    return model


def train_kmeans(name:str='sob_all', n_clusters=3, size=10000):
    
    d2v_model = load_doc2vec(name)
    vectors = d2v_model.dv.get_normed_vectors()[:size]
    t_start = time.time()
    model = KMeans(n_clusters=n_clusters)
    model.fit(vectors)
    m_path = models_path + 'KMeans/kmeans_' + name + f'_n={n_clusters}.pkl'
    joblib.dump(model, m_path)
    print(f'KMeans trained in {round(time.time()-t_start, 1)} sec')
    
    return model


def train_models_vary_clusters(name:str='sob_all'):
    
    d2v_model = load_doc2vec(name)
    vectors = d2v_model.dv.get_normed_vectors()
    
    for n in range(5, 20):
        t_start = time.time()
        model = KMeans(n_clusters=n)
        model.fit(vectors)
        m_path = models_path + 'KMeans/kmeans_' + name + f'_n={n}.pkl'
        joblib.dump(model, m_path)
        print(f'KMeans trained in {round(time.time()-t_start, 1)} sec')


if __name__ == '__main__':
    
    # get_texts()
    # train_doc2vec()
    
    # model = Doc2Vec.load('doc2vec.model')
    # count = 0
    # for i in range(101, 1000):
    #     s = model.dv.most_similar(str(i))
    #     for n in s:
    #         key = int(n[0])
    #         if key <= 100:
    #             print(key)
    #             count += 1
    # print()
    # print(count)
    
    
    pass




# model.infer_vector(['review', 'c']) -> для unseen docs, также можно:
# gensim.models.doc2vec.Doc2Vec.similarity_unseen_docs(model, text1, text2)


