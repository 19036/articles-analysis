import sqlite3
import sys, os, time
import requests
# import urllib.request
# import pypdf
# import tarfile
import traceback
import json  # для конвертации словарь <--> строка: json.dumps(dict), json.loads(str)
import nltk
from nltk.corpus import stopwords
from gensim.models.word2vec import Word2Vec
import pyalex
from threading import Thread, Lock
from pyalex import Works, Authors, Sources, Institutions, Concepts, Publishers, Funders

pyalex.config.email = "g.avilkin@g.nsu.ru"

db_path = '../local_db/articles.db'
logs_path = ('../logs/' + time.ctime().replace(' ', '___') + '.txt').replace(':', '-')
conn_attempt = 1
try:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
except:
    print(f"Couldn't connect to database: {db_path}")
    conn_attempt = 0

""" Создание таблицы """
cursor.execute("""
CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    title TEXT,
    authors TEXT,
    abstract TEXT,
    referenced_works TEXT,
    cites_this_work TEXT,
    cited_by_count INT,
    publication_year INT,
    keywords_oa TEXT,
    topics_oa TEXT,
    level INT
)
""")

lock = Lock()


class Time_Check():
    
    def __init__(self):
        
        self.all = 0
        self.request = 0
        self.db = 0
        self.count = 0
        self.stop = 0
        self.number_of_threads = 1
    
    def clear(self):
        number_of_threads = self.number_of_threads
        self.__init__()
        self.number_of_threads = number_of_threads
    
    # def adreq(self, t):
    #     self.all += t
    #     self.request += t
        
    # def addb(self, t):
    #     self.all += t
    #     self.db += t
    
    def info(self):
        if self.all == 0:
            return 'all_time: 0'
        req_percent = round(self.request/self.all*100,1)
        db_percent = round(self.db/self.all*100,1)
        other_percent = round(100 - req_percent - db_percent, 1)
        speed = round(self.all/self.count/self.number_of_threads, 3)
        s = f'count: {self.count}, speed: {speed} sec/article, request: {req_percent}%, db: {db_percent}, other: {other_percent}%\n'
        return s


time_check = Time_Check()

def database_changing():
    
    " Добавление столбцов "
    # cursor.execute('alter table articles add column cites_this_work')
    
    " Удаление столбцов "
    # cursor.execute('alter table articles drop column id')
    
    " Вставка значений "
    # cursor.execute('update articles set level = ?', (1,))
    
    pass

def sql_requests():
    
    " Количество элементов, удовлетворяющих свойству "
    t_start = time.time()
    cursor.execute('''
                    select id
                    from articles
                    where level = ? and title is ?
                    ''', (1, 'null'))
                   
    ids = cursor.fetchall()
    print(len(ids))
    
    print(f'Time: {round(time.time() - t_start, 1)}')
    return ids
    

def work_parsing(work):
    
    # str: OpenAlexID
    id_ = work['id'].split('/')[-1] 
    
    # str(dict): abstract_inverted_index (word: [indexes of this word])
    abstract_inverted_index = json.dumps(work['abstract_inverted_index'])
    
    # str: title
    title = work['title']
    
    # str: authors ('a1, a2, a3')
    authors = ', '.join(list(map(lambda x: x['author']['display_name'], work['authorships'])))
    
    # str: referenced_works_ids ('id1, id2, id3')
    referenced_works = ', '.join(list(map(lambda x: x.split('/')[-1], work['referenced_works'])))
    
    # int: cited_by_count
    cited_by_count = work['cited_by_count']
    
    # int: publication year
    publication_year = work['publication_year']
    
    # str(dict): keywords_OA (word: score)
    if 'keywords' in work:
        keywords_list = work['keywords']
        keywords = dict()
        for keyword_info in keywords_list:
            keywords[keyword_info['keyword']] = keyword_info['score']
        keywords = json.dumps(keywords)
    else:
        keywords = None
    
    # str(list(dict)): topics_OA (name: t_name, score: score, subfield: sf_name, field: f_name, domain: d_name)
    if 'topics' in work:
        topics_oa = work['topics']
        extract = lambda x: {'name': x['display_name'], 'score': x['score'], 'subfield': x['subfield']['display_name'], 'field': x['field']['display_name'], 'domain': x['domain']['display_name']}
        topics = json.dumps(list(map(extract, topics_oa)))
    else:
        topics = None
        
    column_names = """(id, title, authors, abstract, referenced_works, 
                        cited_by_count, publication_year, keywords_oa, topics_oa)""".replace('\n', '')
    column_content = (id_, title, authors, abstract_inverted_index, referenced_works,
                               cited_by_count, publication_year, keywords, topics)
    
        
    
    return column_names, column_content


def stop_sign():
    sign = input()
    if sign in ['0', 's', 'stop', 'aa']:
        sys.exit()



def download_works_that_cite_this_work(id_, level=1):
    
    global time_check
    # global cursor
    cites_this_work = []
    per_page = 200
    page = 1
    while page <= 10000:
        
        url = f'https://api.openalex.org/works?filter=cites:{id_}&per_page={per_page}&page={page}'
        
        t_start = time.time()
        
        t = requests.get(url)
        try:
            t = t.json()
        except:
            with lock:
                with open('D:\\opd/logs/time_distribution.txt', 'a') as f:
                    s = f'problem with url: {url}\nresponse_text: {t.text}\n'
                    print(s)
                    f.write(s)
            continue
        with lock:
            time_check.request += time.time() - t_start
        
        if 'error' in t:
            print(t)
            break
        else:
            if len(t['results']) == 0:
                break
            else:
                page += 1
                for work in t['results']:
                    temp_id = work['id'].split('/')[-1]
                    cites_this_work.append(temp_id)
                    
                    t_start = time.time()
                    with lock:
                        """ Проверяем, есть ли эта статья в БД """
                        cursor = conn.cursor()
                        cursor.execute('''
                                        SELECT
                                        EXISTS
                                        (select id
                                        from articles
                                        where id == (?))
                                        ''', (temp_id,))
                        already_exists = cursor.fetchall()[0][0]
                        time_check.db += time.time() - t_start
                    if already_exists:
                        # print(count)
                        # count += 1
                        continue
                    
                    column_names, content = work_parsing(work)
                    column_names = column_names[:-1] + ', level)'
                    number_of_columns = len(content)
                    question_string = '(' + '?, '*(number_of_columns + 1)
                    question_string = question_string[:-2] + ')'
                    content += (level + 1,)
                    
                    t_start = time.time()
                    with lock:
                        cursor = conn.cursor()
                        cursor.execute(f"""
                                        insert into articles 
                                        {column_names}
                                        values {question_string}""", 
                                        content)
                        conn.commit()
                        time_check.db += time.time() - t_start
    
    cites_this_work = ', '.join(cites_this_work)
    t_start = time.time()
    with lock:
        cursor = conn.cursor()
        cursor.execute('update articles set cites_this_work = ? where id = ?', (cites_this_work, id_))
        conn.commit()   
        time_check.db += time.time() - t_start
                    

                   
def download_works_that_cite_id(ids, level=1):
    
    global time_check
    all_time = time.time()
    
    for id_ in ids:
        
        download_works_that_cite_this_work(id_, level)
        
        with lock:
            time_check.count += 1
            if time_check.stop:
                return
            
            time_check.all += time.time() - all_time
            if time_check.count >= 100:
                time_check.all = round(time_check.all, 1)
                print(time_check.info())
                with open('D:\\opd/logs/time_distribution.txt', 'a') as f:
                    s = '[' + time.ctime() + '] ' + time_check.info() + ''
                    f.write(s)
                if time_check.stop:
                    return
                time_check.clear()  
        all_time = time.time()
        
        
    
def download_works_that_cite_level(level=1):
    
    global time_check 
    
    cursor.execute('''
                    select id
                    from articles
                    where level = ? and cites_this_work is NULL
                    ''', (level,))
                   
    results = cursor.fetchall()
    results = list(map(lambda x: x[0], results))
    print(f'Number of results: {len(results)}')
    
    number_of_threads = 5
    time_check.number_of_threads = number_of_threads
    path = 'D:\\opd/logs/time_distribution.txt'
    with open(path, 'a') as f:
        s = f'\n\n\nStart downloading works_that_cite_this_work in {number_of_threads} thread(s) at [{time.ctime()}]\n\n\n\n'
        f.write(s)
    threads = []
    stopping_thread = Thread(target=stop_sign, name='stopping_thread')
    length = len(results)
    step = round(length/number_of_threads)
    ids_groups = [results[step*i: step*(i+1)] for i in range(number_of_threads)]
    
    
    for i in range(number_of_threads):
        thread = Thread(target=download_works_that_cite_id, args=(ids_groups[i], level), name=f"thread_{i+1}")
        thread.start()
        threads.append(thread)
    stopping_thread.start()
    
    stopping_thread.join()
    for thread in threads:
        thread.join()


def download_work_by_id(id_, level=1):
    
    global db_path
    with lock:
        try:
            conn = sqlite3.connect(db_path, check_same_thread=False)
            cursor = conn.cursor()
        except:
            print(f"Couldn't connect to database: {db_path}")
            return
    
    if id_ == '':
        return
    global time_check
    global logs_path
    with lock:
        t_start = time.time()
        """ Проверяем, есть ли эта статья в БД """
        cursor = conn.cursor()
        cursor.execute('''
                        SELECT
                        EXISTS
                        (select id
                        from articles
                        where id == (?))
                        ''', (id_,))
        already_exists = cursor.fetchall()[0][0]
        time_check.db += time.time() - t_start
        if already_exists:
            # num_already_exists += 1
            return
    t_start = time.time()
    try:
        work = Works()[id_]
    except:
        with lock:
            with open(logs_path, 'a') as f:
                s = f'[{time.ctime()}] Problem with getting article with id = {id_}\n'
                s += traceback.format_exc() + '\n'
                f.write(s)
            conn.commit()
            conn.close()
        return 
        
    with lock:
        time_check.request += time.time() - t_start
    
    column_names, content = work_parsing(work)
    column_names = column_names[:-1] + ', level)'
    number_of_columns = len(content)
    question_string = '(' + '?, '*(number_of_columns + 1)
    question_string = question_string[:-2] + ')'
    content += (level + 1,)
    
    t_start = time.time()
    with lock:
        try:
            # cursor = conn.cursor()
            cursor.execute(f"""
                            insert into articles 
                            {column_names}
                            values {question_string}""", 
                            content)
            conn.commit()
        except:
            with open(logs_path, 'a') as f:
                s = f'[{time.ctime()}] Problem with adding article with id = {id_}\n'
                s += traceback.format_exc() + '\n'
                f.write(s)
            conn.commit()
            conn.close()
        time_check.db += time.time() - t_start
        conn.commit()
        conn.close()
    

def download_works_by_ids_local(ids, level=1):
    
    global time_check
    global logs_path
    all_time = time.time()
    
    for id_ in ids:
        
        download_work_by_id(id_, level)
        
        with lock:
            time_check.count += 1
            
            time_check.all += time.time() - all_time
            if time_check.count >= 500:
                time_check.all = round(time_check.all, 1)
                print(time_check.info())
                with open(logs_path, 'a') as f:
                    s = '[' + time.ctime() + '] ' + time_check.info() + ''
                    f.write(s)
                time_check.clear()  
        all_time = time.time()


def download_works_by_ids_global(ids, level=1):
    
    global time_check
    global logs_path
    number_of_threads = 7
    time_check.number_of_threads = number_of_threads
    with open(logs_path, 'a') as f:
        s = f'\n\n\nStart downloading works_by_ids in {number_of_threads} thread(s) at [{time.ctime()}]\n\n\n\n'
        f.write(s)
    threads = []
    stopping_thread = Thread(target=stop_sign, name='stopping_thread')
    length = len(ids)
    step = round(length/number_of_threads)
    ids_groups = [ids[step*i: step*(i+1)] for i in range(number_of_threads)]
    
    
    for i in range(number_of_threads):
        thread = Thread(target=download_works_by_ids_local, args=(ids_groups[i], level), name=f"thread_{i+1}")
        thread.start()
        threads.append(thread)
    stopping_thread.start()
    
    stopping_thread.join()
    for thread in threads:
        thread.join()
        

def update_ref_works_ids_request(req_path='../requests/sasan.txt', level=1):
    
    global db_path
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
    except:
        print(f"Couldn't connect to database: {db_path}")
        return
    
    with open(req_path, 'r') as f:
        ids_raw = f.read().split(' ')
    
    print('Updating referenced works request..')
    t_start = time.time()
    
    ids = ''
    num_already_exists = 0
    count = 0
    
    for id_ in ids_raw:
        """ Проверяем, есть ли эта статья в БД """
        # cursor = conn.cursor()
        cursor.execute('''
                        SELECT
                        EXISTS
                        (select id
                        from articles
                        where id == (?))
                        ''', (id_,))
        already_exists = cursor.fetchall()[0][0]
        if already_exists or id_ == '':
            num_already_exists += 1
            continue
        ids += id_ + ' '
        count += 1
        if count % 10000 == 0:
            print(f'{count}: {round(time.time()-t_start, 1)} sec')
            t_start = time.time()
    
    conn.commit()
    conn.close()
    
    ids = ids[:-1]
    
    with open(req_path, 'w') as f:
        f.write(ids)
    print(f'From {count} ids {num_already_exists} already exist, so only {ids.count(" ") + 1} left.')
           
              
        
def download_ref_works(name='', level=1):
    
    global logs_path
    req_path = '../requests/'
    # name = 'george'
    # name = 'valsek'
    # name = 'sasan'
    
    if name == '':
        print('Введите ваше имя (george, valsek, sasan)')
        name = input()
        if name not in ['george', 'valsek', 'sasan']:
            print('Ты чё?')
            return
    req_path = req_path + name + '.txt'
    with open(req_path, 'r') as f:
        ids = f.read().split(' ')
    
    update_ref_works_ids_request(req_path, level)
    
    count = 0
    while count <= 10:
        count += 1
        try:
            download_works_by_ids_global(ids, level)
        except:
            s = f'\n\n{traceback.format_exc()}\nRestarting..\n'
            with open(logs_path, 'a') as f:
                f.write(s)
            time.sleep(5)
    

if __name__ == '__main__':
    
    name = ''
    # name = 'george'
    # name = 'valsek'
    # name = 'sasan'
    
    download_ref_works(name)
    
    
    pass
    


if conn_attempt:
    conn.commit()
    conn.close()
