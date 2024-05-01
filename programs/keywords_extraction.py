import sqlite3
import sys, os, time
import json  # для конвертации словарь <--> строка: json.dumps(dict), json.loads(str)
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# from gensim.test.utils import get_tmpfile
from helpfun import get_current_date, abstract_by_inv_index, sql_execute, inv_index_by_abstract
# from sklearn.cluster import Birch
from sklearn.cluster import KMeans
import joblib
import numpy
from statistics import mean
from bokeh.plotting import figure, show, output_file, save
from bokeh.io import curdoc, export_png
from bokeh.palettes import Turbo256
from bokeh.layouts import gridplot, column, row
from bokeh.models import Range1d, Title, TabPanel, Tabs, ColumnDataSource, Legend
from bokeh.models.tools import BoxZoomTool, ResetTool, PanTool
import multiprocessing.dummy
# from numpy.linalg import norm
import traceback
# from nltk.corpus import stopwords
import editdistance
# import io
# import itertools
import networkx as nx
import nltk
from random import random


curdoc().theme = 'dark_minimal'
db_path = '../databases/nsu_articles.db'
# db_path = '../databases/sob_articles.db'
date = get_current_date()
logs_path = f'../logs/keywords_extraction/{date}.txt'
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
    s += message + '\n\n'
    with open(path, 'a') as f:
        f.write(s)
    print(s)


def load_doc2vec(name:str='nsu_level_1'):
    
    m_path = models_path +  'doc2vec_models/dv_' + name + '.model'
    model = Doc2Vec.load(m_path)
    return model


def load_kmeans(name:str='nsu_level_1', n_cl=5, normed=True):
    
    m_path = models_path +  'KMeans/kmeans_' + name + f'_n={n_cl}' +'_alt'*(1 - normed) + '.pkl'
    model = joblib.load(m_path)
    return model


def randrange(numbers):
    index = round(random() * len(numbers))
    return index


class Keywords_Extraction_For_Text_Info:
    
    def __init__(self, type_='pagerank', connect='adj', weight_type='cos_lin', multiple=False,
                 topn=None, use_values=True):
        
        self.type = type_
        self.connect = connect
        self.weight_type = weight_type
        self.multiple = multiple
        self.topn = topn
        self.use_values = use_values
        # self.info = self.__str__()
    
    def info(self):
        return self.__str__()
    
    def __str__(self):
        s = f'type: {self.type}\n'
        s += f'connect: {self.connect}\n'
        s += f'weight_type: {self.weight_type}\n'
        s += f'multiple: {self.multiple}\n'
        s += f'topn: {self.topn}\n'
        s += f'use_values: {self.use_values}\n'
        return s

class Keywords_Extraction_For_Cluster_Info:
    
    def __init__(self, text_to_cluster='avg_vec',
                 type_='pagerank', connect='adj', weight_type='cos_lin', multiple=False,
                 kwt_topn=None, kwt_use_values=True):
        
        self.ttc = text_to_cluster
        self.kwt_info = Keywords_Extraction_For_Text_Info(type_, connect, weight_type, multiple,
                                                          topn=kwt_topn, use_values=kwt_use_values)
        # self.info = self.__str__()
    
    def info(self):
        return self.__str__()
    
    def __str__(self):
        s = f'text to cluster method: {self.ttc}\n'
        s += self.kwt_info.info()
        return s
    
    def for_db(self):
        s = f'ttc_{self.ttc}___'
        s += f'type_{self.kwt_info.type}___'
        s += f'connect_{self.kwt_info.connect}___'
        s += f'weight_type_{self.kwt_info.weight_type}___'
        s += f'multiple_{self.kwt_info.multiple}___'
        s += f'topn_{self.kwt_info.topn}___'
        s += f'use_values_{self.kwt_info.use_values}___'
        return s

class Database_Info:
    
    def __init__(self, name='nsu_level_1'):
        
        self.name = name
        if name == 'nsu_level_1':
            self.db_path = '../databases/nsu_articles.db'
        elif name == 'sob_all':
            self.db_path = '../databases/sob_articles.db'
        else:
            print('unknown name for database')


def extract_ngrams(text:list, num:int):
    if type(text) == str:
        text = text.split()
    n_grams = nltk.ngrams(text, num)
    Grams = [grams for grams in n_grams]
    return Grams
 
   
def keywords_for_text(text:list, model=load_doc2vec('nsu_level_1'), zipp=False,
                connect='adj', weight_type='cos_lin', multiple=False, info=None):
    
    """ 
        zipp = True/False - выдавать ли лишь список первых (5) ключевых слов, без их веса
        connect = all/adj - соединяем все/смежные слова
        weight_type = cos_pos/cos_lin - в качестве веса берём положительную часть косинуса/переводим его в [0, 1] x -> (x+1)/2
        multiple = True/False - если connect == adj, то умножаем ли вес на число встреч пары слов
    
    """
    
    if info is not None:
        connect = info.connect
        weight_type = info.weight_type
        multiple = info.multiple
    
    if type(text) == str:
        text = text.split()
    nodes = list(set(text))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    
    
    if connect == 'adj':
        nodePairs = extract_ngrams(text, 2)
        for pair in nodePairs:
            w1 = pair[0]
            w2 = pair[1]
            # levDistance = editdistance.eval(firstString, secondString)
            try:
                cos_sim = model.wv.similarity(w1, w2)
                if weight_type == 'cos_pos' and cos_sim > 0:
                    weight = cos_sim
                elif weight_type == 'cos_lin':
                    weight = (cos_sim + 1) / 2
                else:
                    continue
                
                if multiple:
                    weight *= nodePairs.count((w1, w2))
                    
                graph.add_edge(w1, w2, weight=weight)
            except:
                # print(w1, w2)
                pass
    elif connect == 'all':
        for w1 in text:
            for w2 in text:
                try:
                    cos_sim = model.wv.similarity(w1, w2)
                    if weight_type == 'cos_pos' and cos_sim > 0:
                        weight = cos_sim
                    elif weight_type == 'cos_lin':
                        weight = (cos_sim + 1) / 2
                    else:
                        continue
                        
                    graph.add_edge(w1, w2, weight=weight)
                except:
                    # print(w1, w2)
                    pass
    
    tol = 0.0001
    n_iter = 1000
    try:
        page_rank = nx.pagerank(graph, weight='weight', tol=tol, max_iter=n_iter)
    except:
        print('FAKAP МАЗАФАКА')
    
    keywords = dict(sorted(page_rank.items(), key=lambda elem: elem[1], reverse=True))
    if zipp:
        keywords = list(keywords.keys())[:5]
    return keywords

class Text:
    
    def __init__(self, text, index=None, type_='nsu_level_1',
                 kws_extr_info=Keywords_Extraction_For_Text_Info()):
        
        self.get_text(text)
        self.textstr = ' '.join(self.text)
        self.index = index
        self.type = type_
        self.kws_extr_info = kws_extr_info
        # Посчитаны ли ключевые слова
        self.kws_calculated = False
    
    def get_text(self, text):
        
        if type(text) == str:
            if len(text) == 0:
                self.text = text
            elif text[0] == '{':
                self.text = abstract_by_inv_index(text).split()
            else:
                self.text = text.split()
        elif type(text) == list:
            self.text = text
        elif type(text) == dict:
            self.text = abstract_by_inv_index(text).split()
    
    def __str__(self):
        return ' '.join(self.text)
    
    def get_keywords(self, dv=None):
        
        if dv is None:
            dv = load_doc2vec(self.type)
        
        self.keywords = keywords_for_text(self.text, model=dv, info=self.kws_extr_info)
        self.kws_calculated = True
        kws_vec = 0
        for item in list(self.keywords.items())[:self.kws_extr_info.topn]:
            kw, value = item
            try:
                locvec = dv.wv[kw]
            except:
                continue
            if self.kws_extr_info.use_values:
                locvec = locvec * value
            kws_vec = kws_vec + locvec
        
        vol = len(list(self.keywords.items())[:self.kws_extr_info.topn])
        if vol > 0 :
            self.kws_vec = kws_vec / vol
        else:
            self.kws_vec = None
            # s = f'Bad text (no keywords), index={self.index}:\n\n'
            # s += self.textstr + '\n\n'
            # write_logs(s)
        return self.keywords
    
                
def get_texts(db_name='nsu_level_1', indexes=None):
    
    t_start = time.time()
    
    if db_name == 'nsu_level_1':
        db_path_loc = '../databases/nsu_articles.db'
    elif db_name == 'sob_all':
        db_path_loc = '../databases/sob_articles.db'
    else:
        print('unknown name for database')
        return
    
    try:
        conn_loc = sqlite3.connect(db_path_loc, check_same_thread=False)
        cursor_loc = conn_loc.cursor()
    except:
        print(f"Couldn't connect to database: {db_path}")
        return
    
    cursor_loc.execute('SELECT cleaned_abstract FROM articles')
    results = cursor_loc.fetchall()
    if indexes is not None:
        ans = [results[i][0] for i in indexes]
    else:
        ans = list(map(lambda x: x[0], results))
    try:
        conn_loc.close()
    except:
        pass
    
    t = round(time.time() - t_start, 1)
    print(f'Got texts in {t} sec')
    return ans
    

class Single_Cluster:
    
    def __init__(self, indexes=[], n=None, db_info=Database_Info(),
                 kws_extr_info=Keywords_Extraction_For_Cluster_Info(),
                 dv=None):
        
        # Индексы (int) статей из данного кластера
        self.indexes = indexes 
        # Номер (индекс) самого кластера в кластеризации
        self.n = n
        # Информация по базе данных
        self.db_info = db_info
        # Информация по методу извлечения ключевых слов
        self.kws_extr_info = kws_extr_info
        # Посчитаны ли ключевые слова
        self.kws_calculated = False
        # Опционально: добавить сами тексты
        self.texts = []
        # Doc2Vec model:
        self.dv = dv
    
    def add_texts(self, all_texts):
        self.texts = [all_texts[i] for i in self.indexes]
    
    def load_dv(self):
        self.dv = load_doc2vec(self.db_info.name)
    
    def update_texts(self):
        
        if len(self.texts) == 0:
            self.texts = get_texts(self.indexes, self.db_info.name)
        
        if self.texts[0].__class__.__name__ != 'Text':
            self.texts = [Text(text=self.texts[i],
                               index=self.indexes[i],
                               type_=self.db_info.name,
                               kws_extr_info=self.kws_extr_info.kwt_info)
                          for i in range(len(self.indexes))]
    
    def calc_keywords(self):
        
        t_start = time.time()
        if self.dv is None:
            self.load_dv()
        self.update_texts()
        processing = multiprocessing.dummy.Pool(10)
        # print('Start')
        processing.map(lambda x: x.get_keywords(self.dv), self.texts)
        # for i in range(len(self.texts)):
            # self.texts[i].get_keywords(self.dv)
        t = round(time.time() - t_start, 1)
        # print(f'Fin: {t}')
        
        if self.kws_extr_info.ttc == 'avg_vec':
            s = 0
            for text in self.texts:
                if text.kws_vec is not None:
                    s = s + text.kws_vec
            self.kws_vec = s / len(self.texts)
            self.keywords = list(map(lambda x: x[0] , self.dv.wv.most_similar(self.kws_vec, topn=10)))
        else:
            print(f'unknown type for text-to-cluster keywords extraction: {self.kws_extr_info.ttc}')
        
        t = round(time.time() - t_start, 1)
        s = f'keywords for cluster {self.n} calculated in {t} sec'
        write_logs(s)


class Clusters:
    
    def __init__(self, n_cl=5, name='nsu_level_1', normed=True,
                 kws_extr_info=Keywords_Extraction_For_Cluster_Info()):
        
        self.n_cl = n_cl
        self.db_info = Database_Info(name)
        self.normed = normed
        self.kws_extr_info = kws_extr_info
        self.update_meta()
    
    def update_meta(self):
        
        self.kmeans = load_kmeans(self.db_info.name, self.n_cl, self.normed)
        clusters_indexes = [[] for _ in range(self.n_cl)]
        for i in range(len(self.kmeans.labels_)):
            clusters_indexes[self.kmeans.labels_[i]].append(i)
        
        self.dv = load_doc2vec(self.db_info.name)
        self.clusters = [Single_Cluster(indexes=clusters_indexes[index],
                                        n=index,
                                        db_info=self.db_info,
                                        kws_extr_info=self.kws_extr_info,
                                        dv=self.dv)
                         for index in range(self.n_cl)]
    
    def fill_clusters(self, all_texts=None):
        
        if all_texts is None:
            all_texts = get_texts(self.db_info.name)
        
        for index in range(self.n_cl):
            self.clusters[index].texts = [all_texts[i] for i in self.clusters[index].indexes]
            # return self.clusters[index]
            self.clusters[index].update_texts()
    
    def calc_keywords(self):
        s = f'Starting keywords extraction for clusters of {self.db_info.name} articles, n_clusters={self.n_cl}' + ', normed'*self.normed + '\n'
        s += self.kws_extr_info.info()
        write_logs(s)
        for index in range(self.n_cl):
            self.clusters[index].calc_keywords()
    
    def __getitem__(self, index):
        return self.clusters[index]
     
    def save_clusters_kws_vectors(self):
        kw_db_path = '../databases/keywords_' + self.db_info.name[:3] + '.db'
        kwconn = sqlite3.connect(kw_db_path, check_same_thread=False)
        kwcursor = kwconn.cursor()   
        try:
            table = f'kws_vectors_for_clusters_{self.n_cl}'
            info = self.kws_extr_info.for_db()
            for i in range(self.n_cl):
                vec = json.dumps(list(map(float, self.clusters[i].kws_vec)))
                try:
                    kwcursor.execute(f'INSERT INTO {table} (id, {info}) VALUES (?, ?)', (i, vec))
                except:
                    try:
                        kwcursor.execute(f'UPDATE {table} SET {info} = ? WHERE id = ?', (vec, i))
                    except:
                        print('Something went wrong with inserting {vec}')
        except:
            print(traceback.format_exc())
        kwconn.commit()
        kwconn.close()
    
    def save_articles_kws_clusters(self):
        
        t_start = time.time()
        kwcl_db_path = '../databases/clusters_by_keywords_'  + self.db_info.name[:3] + '.db'
        kwclconn = sqlite3.connect(kwcl_db_path, check_same_thread=False)
        kwclcursor = kwclconn.cursor()
        try:
            table = f'kws_clusters_for_articles_{self.n_cl}' 
            info = self.kws_extr_info.for_db()
            km = load_kmeans(self.db_info.name, self.n_cl, self.normed)
            for index in range(len(km.labels_)):
                cluster_index = km.labels_[index]
                local_index = self.clusters[cluster_index].indexes.index(index)
                vec = self.clusters[cluster_index].texts[local_index].kws_vec
                if vec is None:
                    cluster_kws_index = 0
                else:
                    try:
                        cosines = list(map(lambda x: 
                                       (x.kws_vec @ vec)/numpy.linalg.norm(vec)/numpy.linalg.norm(x.kws_vec),
                                       self.clusters))
                    except:
                        pass
                    # return cosines
                    max_cos = max(cosines)
                    cluster_kws_index = cosines.index(max_cos)
                try:
                    kwclcursor.execute(f'INSERT INTO {table} (id, {info}) VALUES (?, ?)', (index, cluster_kws_index))
                except:
                    try:
                        kwclcursor.execute(f'UPDATE {table} SET {info} = ? WHERE id = ?', (cluster_kws_index, index))
                    except:
                        print('Something went wrong with inserting {index} article')
                        print(traceback.format_exc())
                        return
                
                if index % 1000 == 0:
                    kwclconn.commit()
                    # t = round(time.time() - t_start, 1)
                    # t_start = time.time()
                    # print(f'{index} articles updated, {t} sec')
        except:
            print(traceback.format_exc())
        
        
        
        
        
        kwclconn.commit()
        kwclconn.close()
        

def keywords_for_clusters(name:str='nsu_level_1', n_cl:int=20):
    
    if name == 'nsu_level_1':
        db_path = '../databases/nsu_articles.db'
    elif name == 'sob_all':
        db_path = '../databases/sob_articles.db'
    else:
        print('AAAA')
        return
    
    rewrite = 0
    filename = f'kw_for_clusters_{name}_{n_cl}.txt'
    if filename in os.listdir('../ml_models/keywords') and not rewrite:
        print(f'{filename} already exists')
        return
    
    
    message = f'Starting keywords extraction for clusters of {name} articles, n_clusters={n_cl}\n\n'
    write_logs(message)
    
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    
    t_start = time.time()
    
    req = 'SELECT cleaned_abstract FROM articles'
    cursor.execute(req)
    results = cursor.fetchall()
    # results = cursor.fetchone()
    # return results[0]
    t = round(time.time() - t_start, 1)
    message = f'Got texts in {t} sec'
    write_logs(message)
    t_start = time.time()
    
    # n_cl = 5
    km = load_kmeans(name=name, n_clusters=n_cl)
    dv = load_doc2vec(name=name)
    clusters = [[] for _ in range(n_cl)]
    for i in range(len(km.labels_)):
        clusters[km.labels_[i]].append(i)
    
    # text = abstract_by_inv_index(results[0]).split()
    
    processing = multiprocessing.dummy.Pool(10)
    texts = processing.map(lambda x: abstract_by_inv_index(x[0]).split(), results)
    
    t = round(time.time() - t_start, 1)
    message = f'Texts decoded in {t} sec'
    write_logs(message)
    t_start = time.time()
    
    cluster_keywords_vectors = [0 for _ in range(n_cl)]
    for i in range(n_cl):
        v = 0
        for index in clusters[i]:
            vol = 5
            kw_dict = keywords_for_text(texts[index])
            if kw_dict is None:
                continue
            keywords = list(kw_dict.keys())[:vol]
            v2 = 0
            for word in keywords:
                try:
                    v2 += dv.wv[word]
                except:
                    pass
            v2 /= vol
            v += v2
        cluster_keywords_vectors[i] = v / len(clusters[i])
        
        t = round(time.time() - t_start, 1)
        message = f'Average keywords vector for {i} cluster calculated in {t} sec'
        write_logs(message)
        t_start = time.time()
        
    
    
    kws = []
    
    for keywords_vector in cluster_keywords_vectors:
        # print(dv.wv.most_similar(keywords_vector))
        kws.append(list(map(lambda x: x[0] , dv.wv.most_similar(keywords_vector, topn=30))))
    
    s = ''
    for i in range(len(kws)):
        s += f'{i} {" ".join(kws[i])}\n\n'        
    
    path = f'../ml_models/keywords/kw_for_clusters_{name}_{n_cl}.txt'
    with open(path, 'w', encoding="utf-8") as f:
        f.write(s)
    
    
    
    message = 'Keywords extraction complete\n\n\n'
    write_logs(message)
    
    
    
    try:
        conn.commit()
        conn.close()
    except:
        pass


def check_keywords_for_articles(name:str='nsu_level_1', n_cl:int=4):
    
    km = load_kmeans(name, n_cl)
    
    if name == 'nsu_level_1':
        db_path = '../databases/nsu_articles.db'
    elif name == 'sob_all':
        db_path = '../databases/sob_articles.db'
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    t_start = time.time()
    
    req = 'SELECT cleaned_abstract, keywords_oa FROM articles'
    cursor.execute(req)
    results = cursor.fetchall()
    t = round(time.time() - t_start, 1)
    print(f'Got texts in {t} sec')
    t_start = time.time()
    
    
    sample = 10
    
    clusters = [[] for _ in range(n_cl)]
    labels = km.labels_
    # vectors = dw_model.dv.get_normed_vectors()
    for i in range(len(labels)):
        clusters[labels[i]].append(i)
    
    kw_path = f'../ml_models/keywords/kw_for_clusters_{name}_{n_cl}.txt'
    with open(kw_path, 'r') as f:
        kws_clusters = f.read().split('\n\n')
    kws_clusters.pop(-1)
    
    path = f'../ml_models/keywords/sample/{name}_{n_cl}.txt'
    t_start = time.time()
    for i in range(n_cl):
        cluster = clusters[i]
        s = kws_clusters[i] + '\n\n\n'
        for _ in range(sample):
            index = randrange(cluster)
            abstract, keywords_oa = results[index]
            abstract = abstract_by_inv_index(abstract)
            keywords_textrank = ' '.join(keywords_for_text(abstract, True))
            s += f'ABSTRACT:\n{abstract}\n'
            s += f'KEYWORDS_TEXTRANK: {keywords_textrank}\n'
            s += f'KEYWORDS_OA: {keywords_oa}\n\n'
        
        with open(path, 'a', encoding="utf-8") as f:
            f.write(s)
        
        t = round(time.time() - t_start, 1)
        print(f'cluster {i}: {t} sec')
            
            
def update_kws_db():
    
    """ Делаем БД с векторами ключевых слов для кластеров для разных методов """
    # kw_db_path = '../databases/keywords_nsu.db'
    # kwconn = sqlite3.connect(kw_db_path, check_same_thread=False)
    # kwcursor = kwconn.cursor()
    # for n in range(2, 101):
    #     kwcursor.execute(f""" 
    #                      CREATE TABLE IF NOT EXISTS kws_vectors_for_clusters_{n} (
    #                      id INT PRIMARY KEY
    #                          )
    #                      """)
    #     ttc = 'avg_vec'
    #     type_ = 'pagerank'
    #     for connect in ['adj', 'all']:
    #         for weight_type in ['cos_pos', 'cos_lin']:
    #             for multiple in [True, False]:
    #                 for topn in [None, 5, 10]:
    #                     for use_values in [True, False]:
    #                         kws_extr_info = Keywords_Extraction_For_Cluster_Info(
    #                             ttc, type_, connect, weight_type, multiple, topn, use_values)
    #                         info = kws_extr_info.for_db()
    #                         # print(info)
    #                         kwcursor.execute(f"""ALTER TABLE kws_vectors_for_clusters_{n} 
    #                                          ADD COLUMN {info} TEXT DEFAULT NULL
    #                                          """)
    
    # kwconn.commit()
    # kwconn.close()
    
    """ Делаем БД с кластеризацией статей по векторам ключевых слов для разных методов """
    kwcl_db_path = '../databases/clusters_by_keywords_nsu.db'
    kwclconn = sqlite3.connect(kwcl_db_path, check_same_thread=False)
    kwclcursor = kwclconn.cursor()
    # for n in range(2, 101):
    #     kwclcursor.execute(f""" 
    #                       CREATE TABLE IF NOT EXISTS kws_clusters_for_articles_{n} (
    #                       id INT PRIMARY KEY
    #                           )
    #                       """)
    #     ttc = 'avg_vec'
    #     type_ = 'pagerank'
    #     for connect in ['adj', 'all']:
    #         for weight_type in ['cos_pos', 'cos_lin']:
    #             for multiple in [True, False]:
    #                 for topn in [None, 5, 10]:
    #                     for use_values in [True, False]:
    #                         kws_extr_info = Keywords_Extraction_For_Cluster_Info(
    #                             ttc, type_, connect, weight_type, multiple, topn, use_values)
    #                         info = kws_extr_info.for_db()
    #                         # print(info)
    #                         kwclcursor.execute(f"""ALTER TABLE kws_clusters_for_articles_{n} 
    #                                           ADD COLUMN {info} INT DEFAULT NULL
    #                                           """)
                                              
    #     # kwclcursor.execute(f"""ALTER TABLE kws_clusters_for_articles_{n} 
    #     #                                       ADD COLUMN kmeans INT DEFAULT NULL
    #     #                                       """)
    
        # km = load_kmeans(n_cl=n)
        # table = f'kws_clusters_for_articles_{n}'
        # for index in range(len(km.labels_)):
        #     cluster = int(km.labels_[index])
        #     # return
        #     try:
        #         kwclcursor.execute(f'INSERT INTO {table} (id, kmeans) VALUES (?, ?)', (index, cluster))
        #     except:
        #         try:
        #             kwclcursor.execute(f'UPDATE {table} SET kmeans = ? WHERE id = ?', (cluster, index))
        #         except:
        #             print('Something went wrong with inserting {index} article')
        #             print(traceback.format_exc())
        #             return
    
    
    kwclconn.commit()
    kwclconn.close()


def already_calculated(n, name, kws_extr_info):
    
    flag = True
    info = kws_extr_info.for_db()
    table_clusters = f'kws_vectors_for_clusters_{n}'
    table_articles = f'kws_clusters_for_articles_{n}'
    kw_db_path = '../databases/keywords_' + name[:3] + '.db'
    kwcl_db_path = '../databases/clusters_by_keywords_' + name[:3] + '.db'
    
    kwconn = sqlite3.connect(kw_db_path, check_same_thread=False)
    kwcursor = kwconn.cursor()
    
    try:
        kwcursor.execute(f'SELECT {info} from {table_clusters}')
        results = kwcursor.fetchall()
        if len(results) < n:
            flag = False
            return flag
        for item in results:
            if item[0] is None:
                flag = False
                return flag
    except:
        print(traceback.format_exc())
    
    kwclconn = sqlite3.connect(kwcl_db_path, check_same_thread=False)
    kwclcursor = kwclconn.cursor()
    
    try:
        kwclcursor.execute(f'SELECT {info} from {table_articles}')
        km = load_kmeans(name=name, n_cl=n)
        results = kwclcursor.fetchall()
        if len(results) < len(km.labels_):
            flag = False
            return flag
        for item in results:
            if item[0] is None:
                flag = False
                return flag
    except:
        print(traceback.format_exc())
    
    kwclconn.commit()
    kwclconn.close()
    
    return flag


def calc_some_shit():
    
    name = 'nsu_level_1'
    ttc = 'avg_vec'
    type_ = 'pagerank'
    connect = 'adj'
    weight_type = 'cos_lin'
    multiple = False
    topn = None
    use_values = True
    kws_extr_info = Keywords_Extraction_For_Cluster_Info(
            ttc, type_, connect, weight_type, multiple, topn, use_values)
    # info = kws_extr_info.for_db()
    n = 5
    
    for n in range(100,101,5):
        for connect in ['adj']:
            for weight_type in ['cos_lin', 'cos_pos']:
                for multiple in [False, True]:
                    for topn in [None, 5, 10]:
                        for use_values in [True, False]:
                            kws_extr_info = Keywords_Extraction_For_Cluster_Info(
                                    ttc, type_, connect, weight_type, multiple, topn, use_values)
                            # info = kws_extr_info.for_db()
                            if already_calculated(n, name, kws_extr_info):
                                print(f'These things for {name}, {n}, {kws_extr_info.info()}ALREADY CALCULATED\n')
                                continue
                            clusters = Clusters(n, name, True, kws_extr_info)
                            clusters.fill_clusters()
                            clusters.calc_keywords()
                            write_logs('Keywords calculated')
                            clusters.save_clusters_kws_vectors()
                            clusters.save_articles_kws_clusters()
                            write_logs('All info saved\n\n')
    

def check_different_textranks_all_clusters():
    
    kwcl_db_path = '../databases/clusters_by_keywords_nsu.db'
    kwclconn = sqlite3.connect(kwcl_db_path, check_same_thread=False)
    kwclcursor = kwclconn.cursor()
    name = 'nsu_level_1'
    ttc = 'avg_vec'
    type_ = 'pagerank'
    connect = 'adj'
    weight_type = 'cos_lin'
    multiple = False
    topn = None
    use_values = True
    # kws_extr_info = Keywords_Extraction_For_Cluster_Info(
    #         ttc, type_, connect, weight_type, multiple, topn, use_values)
    # info = kws_extr_info.for_db()
    
    rows = []
    for connect in ['adj']:
        for weight_type in ['cos_lin', 'cos_pos']:
            for multiple in [False, True]:
                figures = []
                for topn in [None, 5, 10]:
                    for use_values in [True, False]:
                        
                        f = figure(tools="crosshair,pan,wheel_zoom,box_zoom,reset,hover")
                        x_axis = list(range(5, 101, 5))
                        acc_values = []
                        for n in range(5, 101, 5):
                            
                            kws_extr_info = Keywords_Extraction_For_Cluster_Info(
                                    ttc, type_, connect, weight_type, multiple, topn, use_values)
                            if already_calculated(n, name, kws_extr_info):
                                
                                info = kws_extr_info.for_db()
                                
                                table = f'kws_clusters_for_articles_{n}'
                                
                                try:
                                    kwclcursor.execute(f'SELECT {info}, kmeans FROM {table}')
                                    results = kwclcursor.fetchall()
                                except:
                                    print(traceback.format_exc())
                                
                                count = 0
                                for item in results:
                                    if item[0] == item[1]:
                                        count += 1
                                acc = count/len(results)
                                acc_values.append(acc)
                            else:
                                acc_values.append(0)
                                
                        f.vbar(x=x_axis, top=acc_values, width=0.7)
                        title = weight_type[-3:] + ' '
                        title += 'm=' + 'T' * multiple + 'F' * (1 - multiple) + ' '
                        title += 'top' + str(topn) + ' '
                        title += 'u=' + 'T' * use_values + 'F' * (1 - use_values)
                        f.title = title
                        figures.append(f)
                rows.append(figures)
    
    kwclconn.close()
    
    grid = gridplot(rows)
    grid.sizing_mode = 'scale_both'
    grid.rows = str(100//len(rows)) + '%'
    grid.cols = str(100//len(rows[0])) + '%'
    grid.toolbar_location = 'below'
    show(grid)
    
    path = '../ml_models/other/different_textranks_all_clusters.html'
    output_file(path, title='different_textranks')
    save(grid)


def check_different_textranks_for_cluster(n=5):
    
    kwcl_db_path = '../databases/clusters_by_keywords_nsu.db'
    kwclconn = sqlite3.connect(kwcl_db_path, check_same_thread=False)
    kwclcursor = kwclconn.cursor()
    name = 'nsu_level_1'
    ttc = 'avg_vec'
    type_ = 'pagerank'
    connect = 'adj'
    weight_type = 'cos_lin'
    multiple = False
    topn = None
    use_values = True
    # kws_extr_info = Keywords_Extraction_For_Cluster_Info(
    #         ttc, type_, connect, weight_type, multiple, topn, use_values)
    # info = kws_extr_info.for_db()
    
    accs = []
    num = 0
    information = ''
    for connect in ['adj']:
        for weight_type in ['cos_lin', 'cos_pos']:
            for multiple in [False, True]:
                for topn in [None, 5, 10]:
                    for use_values in [True, False]:
                        
                        kws_extr_info = Keywords_Extraction_For_Cluster_Info(
                                ttc, type_, connect, weight_type, multiple, topn, use_values)
                        if already_calculated(n, name, kws_extr_info):
                            
                            info = kws_extr_info.for_db()
                            
                            table = f'kws_clusters_for_articles_{n}'
                            
                            try:
                                kwclcursor.execute(f'SELECT {info}, kmeans FROM {table}')
                                results = kwclcursor.fetchall()
                            except:
                                print(traceback.format_exc())
                            
                            count = 0
                            for item in results:
                                if item[0] == item[1]:
                                    count += 1
                            acc = count/len(results)
                        else:
                            acc = 0
                                
                        accs.append(acc)
                        title = weight_type[-3:] + ' '
                        title += 'm=' + 'T' * multiple + 'F' * (1 - multiple) + ' '
                        title += 'top' + str(topn) + ' '
                        title += 'u=' + 'T' * use_values + 'F' * (1 - use_values)
                        information += f'{num}: ' + title + '\n'
                        num += 1
    
    kwclconn.close()
    x_axis = list(range(len(accs)))
    f = figure(tools="crosshair,pan,wheel_zoom,box_zoom,reset,hover", toolbar_location='below',
               title=f'accuracies for n_cl={n}', sizing_mode='scale_both')
    f.vbar(x=x_axis, top=accs, width=0.7)
    
    info_box = figure(tools="crosshair,pan,wheel_zoom,box_zoom,reset,hover")
    info_box.circle([], [])
    info_box.title = information
    info_box.toolbar_location = None
    gridf = gridplot([[f]])
    gridf.toolbar_location = 'below'
    gridf.sizing_mode = 'scale_both'
    
    grid = gridplot([[gridf, info_box]])
    grid.toolbar_location = None
    grid.sizing_mode = 'scale_both'
    grid.rows = '100%'
    grid.cols = ['80%', '20%']
    # grid = column([f, info_box])
    # grid.sizing_mode = 'scale_both'
    # show(grid)
    
    path = f'../ml_models/other/different_textranks_n_cl={n}.html'
    output_file(path, title=f'different_textranks_for_n_cl={n}')
    save(grid)
    output_file('../../delete_me.html')


def check_textrank_all_clusters():
    
    kwcl_db_path = '../databases/clusters_by_keywords_nsu.db'
    kwclconn = sqlite3.connect(kwcl_db_path, check_same_thread=False)
    kwclcursor = kwclconn.cursor()
    name = 'nsu_level_1'
    ttc = 'avg_vec'
    type_ = 'pagerank'
    connect = 'adj'
    weight_type = 'cos_pos'
    multiple = False
    topn = None
    use_values = False
    # kws_extr_info = Keywords_Extraction_For_Cluster_Info(
    #         ttc, type_, connect, weight_type, multiple, topn, use_values)
    # info = kws_extr_info.for_db()
    
                        
    f = figure(tools="crosshair,pan,wheel_zoom,box_zoom,reset,hover")
    x_axis = list(range(5, 101, 5))
    acc_values = []
    for n in range(5, 101, 5):
        
        kws_extr_info = Keywords_Extraction_For_Cluster_Info(
                ttc, type_, connect, weight_type, multiple, topn, use_values)
        if already_calculated(n, name, kws_extr_info):
            
            info = kws_extr_info.for_db()
            
            table = f'kws_clusters_for_articles_{n}'
            
            try:
                kwclcursor.execute(f'SELECT {info}, kmeans FROM {table}')
                results = kwclcursor.fetchall()
            except:
                print(traceback.format_exc())
            
            count = 0
            for item in results:
                if item[0] == item[1]:
                    count += 1
            acc = count/len(results)
            acc_values.append(acc)
        else:
            acc_values.append(0)
            
    f.vbar(x=x_axis, top=acc_values, width=0.7)
    title = weight_type[-3:] + ' '
    title += 'm=' + 'T' * multiple + 'F' * (1 - multiple) + ' '
    title += 'top' + str(topn) + ' '
    title += 'u=' + 'T' * use_values + 'F' * (1 - use_values)
    f.title = title
    f.height = 500
    f.width = 1000
    # f.sizing_mode = 'stretch_both'
    f.toolbar_location = 'below'
    
    
    kwclconn.close()
    
    path = f'../ml_models/other/textranks_all_clusters_{title}.html'
    output_file(path, title='different_textranks')
    save(f)
    show(f)
    output_file('../../delete_me.html')


# Для тестов, абстракт обзора физики частиц
text = 'abstract review summarize much particle physics cosmology use datum previous edition plus _number_ new measurement _number_ paper list evaluate average measured property gauge boson recently discover higgs boson lepton quark meson baryon summarize search hypothetical particle supersymmetric particle heavy boson axion dark photon etc particle property search limit list summary tables give numerous table figure formulae review topic higgs boson physics supersymmetry grand unified theories neutrino mixing dark energy dark matter cosmology particle detectors colliders probability statistics among _number_ review many new heavily revise include new review high energy soft qcd diffraction one determination ckm angles b hadrons review divide two volume volume _number_ include summary tables _number_ review article volume _number_ consist particle listings contain also _number_ review address specific aspect datum present listings complete review publish online website particle data group journal volume _number_ available print pdg book particle physics booklet summary tables essential table figure equation select review article available print web version optimize use phone well android app'


def rand_graphic():
    
    x_axis = [1, 2, 3, 4, 5]
    y_axis = [random() for _ in range(5)]
    f = figure()
    f.vbar(x=x_axis, top=y_axis)
    f.title = str(random())
    return f


def test():
    
    rows = []
    for i in range(3):
        figures = []
        for j in range(4):
            figures.append(rand_graphic())
        rows.append(figures)
    
    grid = gridplot(rows)
    grid.sizing_mode = 'scale_both'
    grid.rows = str(100//len(rows)) + '%'
    grid.cols = str(100//len(rows[0])) + '%'
    grid.toolbar_location = 'below'
    show(grid)
    
    
    
    pass



if __name__ == '__main__':
    
    
    
    
    pass




try:
    conn.commit()
    conn.close()
except:
    pass