import sqlite3
import sys, os, time
import json  # для конвертации словарь <--> строка: json.dumps(dict), json.loads(str)
import gensim
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from helpfun import get_current_date, abstract_by_inv_index, sql_execute
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
import joblib
from statistics import mean, stdev
from bokeh.plotting import figure, show, output_file, save
from bokeh.io import curdoc, export_png
from bokeh.palettes import Turbo256
from bokeh.layouts import gridplot, column, row
from bokeh.models import Range1d, Title, TabPanel, Tabs, ColumnDataSource, Legend
from bokeh.models.tools import BoxZoomTool, ResetTool, PanTool
import multiprocessing.dummy
from numpy.linalg import norm
import numpy
import traceback
from nltk.corpus import stopwords
import editdistance
import io
import itertools
import networkx as nx
import nltk


curdoc().theme = 'dark_minimal'
# db_path = '../databases/articles.db'
db_path = '../databases/sob_articles.db'
date = get_current_date()
logs_path = f'../logs/ml_models/{date}.txt'
train_data_path = '../ml_models/train_data/'
models_path = '../ml_models/'

try:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
except:
    print(f"Couldn't connect to database: {db_path}")

def write_logs(message:str, path:str=logs_path):
    
    """ Пишет логи (в файл с текущей датой)
        Используется в: working_with_database
                        ml_models
    """
    
    if path is None:
        # try:
        #     global logs_path
        #     path = logs_path
        # except:
        path = '../logs/unknown_logs.txt'
    
    if path[-4:] != '.txt':
        path += '_unknown.txt'
        
    s = f'[{time.ctime()}]\n'
    s += message + '\n\n\n'
    with open(path, 'a') as f:
        f.write(s)
    print(s)

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
    
    
def load_doc2vec(name:str='nsu_level_1'):
    
    m_path = models_path +  'doc2vec_models/dv_' + name + '.model'
    model = Doc2Vec.load(m_path)
    return model

def load_birch(name:str='nsu_level_1', n_clusters=5):
    
    m_path = models_path +  'birch/birch_' + name + f'_n={n_clusters}.pkl'
    model = joblib.load(m_path)
    return model

def load_kmeans(name:str='nsu_level_1', n_clusters=5, alt=False):
    
    m_path = models_path +  'KMeans/kmeans_' + name + f'_n={n_clusters}' +'_alt'*alt + '.pkl'
    model = joblib.load(m_path)
    return model


def get_norms(name='nsu_level_1'):
    ' Считает нормы векторов для текстов по Doc2Vec и записывает в файл'
    model = load_doc2vec(name)
    vectors = model.dv.vectors
    norms = list(map(lambda x: float(norm(x)), vectors))
    norms = sorted(norms)
    s = json.dumps(norms)
    path = models_path + f'other/dv_norm_{name}.txt'
    with open(path, 'w') as f:
        f.write(s)
        

def train_birch(name:str='nsu_level_1', n_clusters=3, size=10000):
    
    d2v_model = load_doc2vec(name)
    vectors = d2v_model.dv.get_normed_vectors()[:size]
    t_start = time.time()
    model = Birch(n_clusters=n_clusters)
    model.fit(vectors)
    m_path = models_path + 'birch/birch_' + name + f'_n={n_clusters}.pkl'
    joblib.dump(model, m_path)
    print(f'Birch trained in {round(time.time()-t_start, 1)} sec')
    
    return model


def train_kmeans(name:str='nsu_level_1', n_clusters=3, alt=False):
    
    filename = f'kmeans_{name}_n={n_clusters}' + '_alt' * alt + '.pkl'
    if filename in os.listdir('../ml_models/KMeans'):
        print(f'{filename} already exists')
        return
    d2v_model = load_doc2vec(name)
    if alt:
        vectors = d2v_model.dv.vectors
    else:
        vectors = d2v_model.dv.get_normed_vectors()
    t_start = time.time()
    model = KMeans(n_clusters=n_clusters, n_init='auto')
    model.fit(vectors)
    m_path = models_path + 'KMeans/kmeans_' + name + f'_n={n_clusters}'+ '_alt' * alt + '.pkl'
    joblib.dump(model, m_path)
    t = round(time.time() - t_start, 1)
    s = f'KMeans for {name}, n_clusters = {n_clusters} trained in {t} sec\n\n\n'
    write_logs(s)


def train_models_vary_clusters(name:str='nsu_level_1'):
    
    for n in range(5, 101, 5):
        train_kmeans(name, n, alt=True)


def calc_distances_for_d2v_vectors(name:str='nsu_level_1', normed=True):
    
    model = load_doc2vec(name)
    vectors = model.dv.get_normed_vectors() if normed else model.dv.vectors
    # db_path = f'../databases/distances/distances_{name}' + '_normed'*normed + '.db'
    db_path = f'D:\\opd/distances/distances_{name}' + '_normed'*normed + '_new.db'
    l_conn = sqlite3.connect(db_path, check_same_thread=False)
    l_cursor = l_conn.cursor()
    l_cursor.execute('''
    CREATE TABLE IF NOT EXISTS distances (
    id INTEGER PRIMARY KEY,
    distances TEXT
    )
    ''')
    s = f'Start distance calculating for {name} ' + 'normed'*normed
    write_logs(s)
    t_start = time.time()
    # for i in range(model.corpus_count):
    for i in [40892, 47405]:
        # req = f'SELECT EXISTS (SELECT id FROM distances WHERE id = {i})'
        # res = l_cursor.execute(req)
        # exists = res.fetchall()[0][0]
        # if not exists:
        if True:
            dists = []
            # l_cursor.execute(f'DELETE FROM distances WHERE id = {i}')
            processing = multiprocessing.dummy.Pool(10)
            vector = vectors[i]
            rad_vectors = [vector - v for v in vectors]
            dists = processing.map(lambda x: round(float(norm(x)), 6), rad_vectors)
            dists = json.dumps(dists)
            req = f'INSERT INTO distances VALUES {(i, dists)}'
            # req = 'UPDATE distances SET distances = ? where id = ?'
            l_cursor.execute(req)
            l_conn.commit()
            if i != 0 and i % 1000 == 0:
                t = round(time.time() - t_start, 1)
                t_start = time.time()
                s = f'{i} calculated, last 1000 in {t} sec'
                write_logs(s)
    
    l_conn.close()
    

def cut_mantissa(name:str='nsu_level_1', normed=True):
    
    db_path = f'../databases/distances/distances_{name}' + '_normed'*normed + '.db'
    l_conn = sqlite3.connect(db_path, check_same_thread=False)
    l_cursor = l_conn.cursor()
    request = 'SELECT COUNT(*) FROM distances'
    l_cursor.execute(request)
    num = l_cursor.fetchall()[0][0]
    t_start = time.time()
    for i in range(num):
        req = f'SELECT distances FROM distances WHERE id = {i}'
        res = l_cursor.execute(req)
        dists = res.fetchall()[0][0]
        dists = json.loads(dists)
        return dists
        processing = multiprocessing.dummy.Pool(10)
        dists = processing.map(lambda x: round(x, 6), dists)
        dists = json.dumps(dists)
        req = f'UPDATE distances SET distances = ? WHERE id = {i}'
        res = l_cursor.execute(req, (dists,))
        l_conn.commit()
        if i != 0 and i % 1000 == 0:
            t = round(time.time() - t_start, 1)
            t_start = time.time()
            s = f'{i} cutted, last 1000 in {t} sec'
            print(s)
    
    l_conn.close()

def get_dists(id_:int=0, name:str='nsu_level_1', normed=False):

    db_path = f'../databases/distances/distances_{name}' + '_normed'*normed + '.db'
    l_conn = sqlite3.connect(db_path, check_same_thread=False)
    l_cursor = l_conn.cursor()
    req = f'SELECT distances FROM distances WHERE id = {id_}'
    l_cursor.execute(req)
    res = l_cursor.fetchall()[0][0]
    return res
    l_conn.close()

def calc_silh_values_for_clustering(n_clusters:int=20, name:str='nsu_level_1', normed=False):
    
    filename = f'kmeans_{name}_n={n_clusters}' + '_normed'*normed + '.txt'
    if filename in os.listdir(f'../ml_models/sv/{name}'):
        print(f'{filename} already exists')
        return
    t_start = time.time()
    s = f'Start calculating silhouette values for {name}, n_clusters={n_clusters}' + ', normed'*normed + '\n\n'
    print(s)
    db_path = f'D:\\opd/distances/distances_{name}' + '_normed'*normed + '.db'
    l_conn = sqlite3.connect(db_path, check_same_thread=False)
    l_cursor = l_conn.cursor()
    model = load_kmeans(name, n_clusters, alt=not normed)
    
    clusters = [[] for _ in range(n_clusters)]
    labels = model.labels_
    # vectors = dw_model.dv.get_normed_vectors()
    for i in range(len(labels)):
        clusters[labels[i]].append(i)
    
    silh_values = []
    
    # m = 100
    # for j in range(n):
    #     nr = norm( vectors[i] - model_K.cluster_centers_[j] )
    #     if m > nr and j != lb:
    #         m = nr
    #         clust_neigh = j
    cluster_neighbours = [-1 for _ in range(n_clusters)]
    for i in range(n_clusters):
        cl_center = model.cluster_centers_[i]
        cl_centers_diff = [cl_center - center for center in model.cluster_centers_]
        distances = list(map(lambda x: float(norm(x)), cl_centers_diff))
        neighbours = [[j, dist] for j, dist in enumerate(distances)]
        neighbours = sorted(neighbours, key=lambda x: x[1])
        cluster_neighbours[i] = neighbours[1][0]
    
    t = round(time.time() - t_start, 1)
    print(f'Preliminaries: {t} sec')
    t_start = time.time()
    for i in range(len(labels)):
        lb = labels[i]
        
        l_cursor.execute(f'''SELECT distances FROM distances WHERE id = {i}''')
        dists = l_cursor.fetchall()[0][0]
        dists = json.loads(dists)
        
        # Среднее расстояние до векторов из данного кластера
        vec_dists = []
        for index in clusters[lb]:
            vec_dists.append(dists[index])
        a = mean(vec_dists)
        
        # Среднее расстояние до векторов из соседнего кластера
        vec_dis = []
        for index in clusters[cluster_neighbours[lb]]:
            vec_dis.append(dists[index])
        b = mean(vec_dis)
        sv = (b - a) / max(a, b)
        silh_values.append(sv)
        
        if i != 0 and i % 1000 == 0:
            t = round(time.time() - t_start, 1)
            s += f'{i}: {t} sec\n'
            print(f'{i}: {t} sec')
            t_start = time.time()
            # break
        
    l_conn.close()
    
    silh_values_by_clusters = [[] for _ in range(n_clusters)]
    for i in range(n_clusters):
        svs = [silh_values[j] for j in clusters[i]]
        # for j in clusters[i]:
        #     if j < 10:
        #         print(j)
        #         print(silh_values[j])
        # return svs
        silh_values_by_clusters[i] = sorted(svs, reverse=True)
    
    t = round(time.time() - t_start, 1)
    s += f'Sorted in {t} sec\n\n\n\n\n'
    print(f'Sorted in {t} sec')
    
    write_logs(s)
    path = f'../ml_models/sv/{name}/kmeans_{name}_n={n_clusters}' + '_normed'*normed + '.txt'
    with open(path, 'w') as f:
        f.write(json.dumps(silh_values_by_clusters))

def compile_silh_values():
    
    # for n in range(2, 101, 5):
    for n in range(5, 101, 5):
        calc_silh_values_for_clustering(n_clusters=n, name='nsu_level_1', normed=False)


def graphic(n_clusters:int=20, name:str='nsu_level_1', normed=False):
    
    filename = f'kmeans_{name}_n={n_clusters}' + '_normed'*normed + '.txt'
    if filename not in os.listdir(f'../ml_models/sv/{name}'):
        print(f'{filename} not exists')
        return
    
    path = f'../ml_models/sv/{name}/kmeans_{name}_n={n_clusters}' + '_normed'*normed + '.txt'
    with open(path, 'r') as f:
        sv = json.loads(f.read())
    
    f = figure(tools="crosshair,pan,wheel_zoom,box_zoom,reset,hover")
    x_axises = []
    count = 0
    for i in range(n_clusters):
        l = len(sv[i])
        x_axises.append([count + j for j in range(l)])
        count += l
        f.vbar(x=x_axises[-1], top=sv[i], width=0.7, color=Turbo256[20 + i * (256-20)//n_clusters])
    
    # all_svs = []
    for i in range(n_clusters):
        # all_svs.extend(sv[i])
        avg = mean(sv[i])
        f.line(x_axises[i], avg, color='white')
    
    
    # show(f)
    path = path[:-4] + '.html'
    
    output_file(path, title='Silhouette values')
    save(f)


def avg_svs(name:str='nsu_level_1', normed=True):
    
    ns = []
    averages = []
    for n in range(1, 101):
        filename = f'kmeans_{name}_n={n}' + '_normed'*normed + '.txt'
        if filename not in os.listdir(f'../ml_models/sv/{name}'):
            continue
        path = f'../ml_models/sv/{name}/kmeans_{name}_n={n}' + '_normed'*normed +'.txt'
        with open(path, 'r') as f:
            sv = json.loads(f.read())
        ns.append(n)
        svs = []
        for svl in sv:
            svs.extend(svl)
        averages.append(mean(svs))
    
    f = figure(tools="crosshair,pan,wheel_zoom,box_zoom,reset,hover")
    f.line(x=ns, y=averages, color='red')
    path = f'../ml_models/sv/{name}/average_{name}' + '_normed'*normed + '.html'
    
    output_file(path, title='Average silhouette values')
    save(f)


def avg_both_svs():
    
    graphic = figure(tools="crosshair,pan,wheel_zoom,box_zoom,reset,hover")
    
    name = 'nsu_level_1'
    ns = []
    averages = []
    for n in range(1, 101):
        filename = f'kmeans_{name}_n={n}' + '_normed' + '.txt'
        if filename not in os.listdir(f'../ml_models/sv/{name}'):
            continue
        path = f'../ml_models/sv/{name}/kmeans_{name}_n={n}' + '_normed' +'.txt'
        with open(path, 'r') as f:
            sv = json.loads(f.read())
        ns.append(n)
        svs = []
        for svl in sv:
            svs.extend(svl)
        averages.append(mean(svs))
    
    graphic.line(x=ns, y=averages, color='yellow', legend_label=name)
    
    name = 'sob_all'
    ns = []
    averages = []
    for n in range(1, 101):
        filename = f'kmeans_{name}_n={n}' + '_normed' + '.txt'
        if filename not in os.listdir(f'../ml_models/sv/{name}'):
            continue
        path = f'../ml_models/sv/{name}/kmeans_{name}_n={n}' + '_normed' +'.txt'
        with open(path, 'r') as f:
            sv = json.loads(f.read())
        ns.append(n)
        svs = []
        for svl in sv:
            svs.extend(svl)
        averages.append(mean(svs))
    
    # f = figure(tools="crosshair,pan,wheel_zoom,box_zoom,reset,hover")
    graphic.line(x=ns, y=averages, color='red', legend_label=name)
    graphic.legend.location = 'top_left'
    
    path = f'../ml_models/sv/average_{name}' + '_both' + '.html'
    
    output_file(path, title='Average silhouette values')
    save(graphic)

def inertias(name:str='nsu_level_1'):
    
    x_axis = []
    inert = []
    for i in range(2, 101):
        model = load_kmeans(name, i)
        x_axis.append(i)
        inert.append(model.inertia_)
    
    f = figure(tools="crosshair,pan,wheel_zoom,box_zoom,reset,hover")
    f.line(x=x_axis, y=inert, color='red')
    path = f'../ml_models/other/inertias_{name}' + '.html'
    
    output_file(path, title='Inertias for KMeans')
    save(f)







def test():
    
    for n in range(5, 101, 5):
        graphic(n, name='nsu_level_1', normed=False)
    
    
    # normed = 0
    # db_path = 'D:\\opd/distances/distances_sob_all' + '_normed'*normed + '.db'
    # l_conn = sqlite3.connect(db_path, check_same_thread=False)
    # l_cursor = l_conn.cursor()
    # t_start = time.time()
    # for i in range(50500):
    #     try:
    #         l_cursor.execute(f'SELECT distances FROM distances WHERE id = {i}')
    #         res = l_cursor.fetchall()
    #     except:
    #         print(i)
    #         continue
    #     if i % 1000 == 0:
    #         t = round(time.time() - t_start, 1)
    #         print(f'{i}: in {t} sec')
    #         t_start = time.time()
    # print(len(res))
    # l_conn.close()
    
    pass


if __name__ == '__main__':
    
    
    
    
    pass




# model.infer_vector(['review', 'c']) -> для unseen docs, также можно:
# gensim.models.doc2vec.Doc2Vec.similarity_unseen_docs(model, text1, text2)


