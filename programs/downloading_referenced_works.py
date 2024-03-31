import sqlite3
import sys, time
import traceback
import json  # для конвертации словарь <--> строка: json.dumps(dict), json.loads(str)
import pyalex
from threading import Thread, Lock
from pyalex import Works

pyalex.config.email = "a.khramov@g.nsu.ru"

db_path = '../local_db/articles.db'
main_db_path = '../../main_db/articles.db'
logs_path = ('../logs/download_ref/' + time.ctime().replace(' ', '___') + '_ref_download.txt').replace(':', '-')
requests_path = '../requests/ref_works_'
number_of_threads = 4
try:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
except:
    print(f"Couldn't connect to database: {db_path}")

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
    cleaned_abstract TEXT
)
""")

lock = Lock()

def write_logs(message:str, time_data=True):
    
    global logs_path
    s = ''
    if time_data:
        s = f'[{time.ctime()}]   '
    s += message + '\n'
    with open(logs_path, 'a') as f:
        f.write(s)
    print(s)

class Time_Check():
    
    def __init__(self):
        
        self.all = 0
        self.request = 0
        self.db = 0
        self.count = 0
        self.number_of_threads = 1
    
    def clear(self):
        number_of_threads = self.number_of_threads
        self.__init__()
        self.number_of_threads = number_of_threads
    
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


def compile_ref_works_ids_request(level=1):
    
    global main_db_path
    global requests_path
    req_path = requests_path + f'level_{level}_block_'
    sure = 0
    # sure = 1
    if not sure:
        print('Are you sure??')
        ans = input()
        if ans not in ['1', 'y', 'yes', 'a']:
            return
    
    try:
        conn = sqlite3.connect(main_db_path, check_same_thread=False)
        cursor = conn.cursor()
    except:
        print(f"Couldn't connect to database: {db_path}")
    
    " Количество элементов, удовлетворяющих свойству "
    t_start = time.time()
    cursor.execute('''
                    SELECT referenced_works
                    FROM articles
                    WHERE level = ?
                    ''', (level,))
                    
    ref_lists = cursor.fetchall()
    print(f'Results: {len(ref_lists)}')
    print(f'Time: {round(time.time() - t_start, 1)}')
    print('Start checking')
    num_already_exists = 0
    count_global = 1
    count_local = 0
    t_start = time.time()
    ids_raw = []
    for ref_list in ref_lists:
        ids_raw += ref_list[0].split(', ')
    
    ids_raw = list(set(ids_raw))
    ids = ''
    for id_ in ids_raw:
        """ Проверяем, есть ли эта статья в БД """
        # cursor = conn.cursor()
        cursor.execute('''
                        SELECT
                        EXISTS
                        (SELECT id
                        FROM articles
                        WHERE id == (?))
                        ''', (id_,))
        already_exists = cursor.fetchall()[0][0]
        time_check.db += time.time() - t_start
        if already_exists or id_ == '':
            num_already_exists += 1
            continue
        count_local += 1
        ids += id_ + ' '
        if count_local >= 100000:
            path = f'{req_path}{count_global}.txt'
            with open(path, 'a') as f:
                f.write(ids[:-1])
            ids = ''
            count_local = 0
            print(f'{count_global} block compiled')
            count_global += 1
    path = f'{req_path}{count_global}.txt'
    with open(path, 'a') as f:
        f.write(ids[:-1])
    if count_global == 1:
        print(f'{count_global} block compiled from {count_local} ids')
    else:
        print(f'Last ({count_global}) block compiled from {count_local} ids')
    print(f'num_already_exists = {num_already_exists}')
    
    try:
        conn.commit()
        conn.close()
    except:
        pass


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
            s = f'Problem with getting article with id = {id_}\n'
            s += traceback.format_exc() + '\n'
            write_logs(s)
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
            s = f'Problem with adding article with id = {id_}\n'
            s += traceback.format_exc() + '\n'
            write_logs(s)
            try:
                conn.commit()
                conn.close()
            except:
                pass
        time_check.db += time.time() - t_start
        try:
            conn.commit()
            conn.close()
        except:
            pass
    

def download_works_by_ids_local(ids, level=1):
    
    global time_check
    all_time = time.time()
    
    for id_ in ids:
        
        download_work_by_id(id_, level)
        
        with lock:
            time_check.count += 1
            
            time_check.all += time.time() - all_time
            if time_check.count >= 500:
                time_check.all = round(time_check.all, 1)
                s = time_check.info()
                write_logs(s)
                time_check.clear()  
        all_time = time.time()


def download_works_by_ids_global(ids, level=1):
    
    global time_check
    global number_of_threads
    time_check.number_of_threads = number_of_threads
    s = f'Start downloading works_by_ids for level_{level} in {number_of_threads} thread(s)\n'
    write_logs(s)
    threads = []
    length = len(ids)
    step = round(length/number_of_threads)
    ids_groups = [ids[step*i: step*(i+1)] for i in range(number_of_threads)]
    
    
    for i in range(number_of_threads):
        thread = Thread(target=download_works_by_ids_local, args=(ids_groups[i], level), name=f"thread_{i+1}")
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
        

def update_ref_works_ids_request(req_path='../requests/ref_works_block_1.txt', level=1):
    
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
            count += 1
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
    
    # req_path = '../requests/'
    # name = 'george'
    # name = 'valsek'
    # name = 'sasan'
    
    # if name == '':
    #     print('Введите ваше имя (george, valsek, sasan)')
    #     name = input()
    #     if name not in ['george', 'valsek', 'sasan']:
    #         print('Ты чё?')
    #         return
    # req_path = req_path + name + '.txt'
    req_path = '../requests/ref_works_block_1.txt'
    
    update_ref_works_ids_request(req_path, level)
    
    with open(req_path, 'r') as f:
        ids = f.read().split(' ')
    
    count = 0
    while count <= 10:
        count += 1
        try:
            download_works_by_ids_global(ids, level)
        except:
            s = f'{traceback.format_exc()}\nRestarting..\n'
            write_logs(s)
    
    

if __name__ == '__main__':
    
    name = ''
    
    # download_ref_works(name)
    
    # compile_ref_works_ids_request(2)
    
    
    pass
    


try:
    conn.commit()
    conn.close()
except:
    pass
