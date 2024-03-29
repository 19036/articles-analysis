import sqlite3
import sys, time
import requests
import json  # для конвертации словарь <--> строка: json.dumps(dict), json.loads(str)
import pyalex
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
        s = f'[{time.ctime()}] count: {self.count}, speed: {speed} sec/article, request: {req_percent}%, db: {db_percent}, other: {other_percent}%\n'
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
    global time_check
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



  

def test(s='sample_text'):
    
    # url = 'https://api.openalex.org/works?filter=cites:W2741809807'
    # url = 'https://api.openalex.org/works/W2741809807'
    # url = f'https://api.openalex.org/works?filter=institutions.id:{nsu_id}&per_page=200&page=2'
    
    # t = requests.get(url).json()
    
    # return t.json()
    
    # pager = Works().filter(cites='W2741809807').paginate(per_page=200)
    
    
    # path = 'D:\\opd/logs/test.txt'
    # with open(path, 'a') as f:
    #     f.write(s + '\n')
    

    pass
       


    
     

if __name__ == '__main__':
    
    
    
    pass
    
    




def archive():
    
    """ Разные куски функций, которые ещё пригодятся """
    
    
    """ Сохраняет pdf - файл по данной url """
    
    # url = 'https://arxiv.org/pdf/2402.00001.pdf'
    # urllib.request.urlretrieve(url, 'testpd.pdf')
    
    
    """ Читает pdf файл, можно вытащить весь текст """
    
    # reader = pypdf.PdfReader('testpd.pdf')
    # print(reader.pages[1].extract_text())
    
    
    """ Проходится по статьям из тестовых 2000 (щас - по первым 20 из них), 
        качает их как pdf, считывает текст, записывает в БД, удаляет pdf """
    
    # obj = html_main_obj()
    
    # divs = obj.find_all("a", {'title': "Abstract"})
    
    # for index in range(20):
    #     div = divs[index]
    #     # url = 'https://arxiv.org' + str(div).split('"')[1]
        
    #     # local_obj = html_obj_by_url(url)
    #     url = 'https://arxiv.org/pdf/' + str(div).split('"')[1][5:] + '.pdf'
    #     urllib.request.urlretrieve(url, 'temp.pdf')
    #     reader = pypdf.PdfReader('temp.pdf')
    #     text = ''
    #     for page in reader.pages:
    #         text += page.extract_text()
        
    #     cursor.execute('update articles_test set text = ? where id = ?', (text, index + 1))
    
    
    """ Некоторые команды курсора sqlite3 """
    
    # cursor.execute('alter table articles rename column mane to title')
        
    # cursor.execute('create index idx_email on users (email)')

    # cursor.execute('insert into users (name, age, email) values (?, ?, ?)', ('who', None, 'hramov@'))

    # cursor.execute('update users set age = ? where id = ?', (21, 4))

    # cursor.execute('delete from articles where id = ?', (4,))

    # cursor.execute('''
    #                select id, name, age 
    #                from users
    #                where age is null
    #                ''')
                   
    # users = cursor.fetchall()
    # for user in users:
    #     print(user)
    
    pass



try:
    conn.commit()
    conn.close()
except:
    pass
