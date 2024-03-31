import sqlite3
import sys, time
import requests
import json  # для конвертации словарь <--> строка: json.dumps(dict), json.loads(str)
import pyalex
import traceback
from threading import Thread, Lock

pyalex.config.email = "a.khramov@g.nsu.ru"

db_path = '../../main_db/articles.db'
logs_path = ('../logs/download_cite/' + time.ctime().replace(' ', '___') + '_cite_download.txt').replace(':', '-')
number_of_threads = 4
try:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
except:
    print(f"Couldn't connect to database: {db_path}")
    
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
        self.downloaded = 0
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
        s = f'count: {self.count}, downloaded: {self.downloaded}, speed (check cites): {speed} sec/article, request: {req_percent}%, db: {db_percent}, other: {other_percent}%\n'
        return s


time_check = Time_Check()


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


def download_works_that_cite_this_work(id_, level=1):
    
    global time_check
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
                s = f'Problem with url: {url}\nresponse_text: {t.text}\n'
                s += f'\nERROR:\n\n{traceback.format_exc()}\n'
                write_logs(s)
            continue
        with lock:
            time_check.request += time.time() - t_start
        
        if 'error' in t:
            s = f'Error in response: {t}'
            write_logs(s)
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
                        try:
                            cursor = conn.cursor()
                            cursor.execute(f"""
                                            insert into articles 
                                            {column_names}
                                            values {question_string}""", 
                                            content)
                            conn.commit()
                            time_check.downloaded += 1
                        except:
                            s = f'Problem with inserting article with id: {content[0]}'
                            s += f'\nERROR:\n\n{traceback.format_exc()}\n'
                            write_logs(s)
                        time_check.db += time.time() - t_start
                        
    
    cites_this_work = ', '.join(cites_this_work)
    t_start = time.time()
    with lock:
        cursor = conn.cursor()
        cursor.execute('update articles set cites_this_work = ? where id = ?', (cites_this_work, id_))
        conn.commit()   
        time_check.db += time.time() - t_start
                    

                   
def download_works_that_cite_id_local(ids, level=1):
    
    global time_check
    all_time = time.time()
    
    for id_ in ids:
        
        download_works_that_cite_this_work(id_, level)
        
        with lock:
            time_check.count += 1
            
            time_check.all += time.time() - all_time
            if time_check.count >= 10:
                time_check.all = round(time_check.all, 1)
                s = time_check.info()
                write_logs(s)
                if time_check.stop:
                    return
                time_check.clear()  
        all_time = time.time()
        
        
    
def download_works_that_cite_id_global(level=1):
    
    global time_check
    global number_of_threads
    global logs_path
    time_check.number_of_threads = number_of_threads
    s = f'Start downloading cite_works for level_{level} in {number_of_threads} thread(s)\n'
    write_logs(s)
    
    cursor.execute('''
                    select id, cited_by_count
                    from articles
                    where level = ? and cites_this_work is NULL
                    ''', (level,))
                   
    results = cursor.fetchall()
    ids = list(map(lambda x: x[0], results))
    cited_by_count = list(map(lambda x: x[1], results))
    s = f'Number of articles to check: {len(results)}'
    write_logs(s)
    s = f'Number of cite works estimation: {sum(cited_by_count)}'
    write_logs(s)
    
    threads = []
    length = len(ids)
    step = round(length/number_of_threads)
    ids_groups = [ids[step*i: step*(i+1)] for i in range(number_of_threads)]
    
    for i in range(number_of_threads):
        thread = Thread(target=download_works_that_cite_id_local, args=(ids_groups[i], level), name=f"thread_{i+1}")
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()



if __name__ == '__main__':
    
    
    # download_works_that_cite_id_global(2)
    
    pass
    
    







try:
    conn.commit()
    conn.close()
except:
    pass
