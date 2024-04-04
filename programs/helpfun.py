import sqlite3
import json
import sys, time
import traceback
from pyalex import Works

# db_path = '../databases/sob_articles.db'
# db_path = '../databases/articles.db'
db_path = '../../delete_me'
logs_path = ('../logs/download_cite/' + time.ctime().replace(' ', '___') + '_cite_download.txt').replace(':', '-')

nsu_id = 'i188973947'
sob_id = 'i4210096862'

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
    level INT,
    cleaned_abstract TEXT
)
""")



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



def download_sob_articles():
    
    """ Скачивание статей с авторами из ИМСОРАН (7 тыщ) """
    
    
    pager = Works().filter(institutions={'id': sob_id}).paginate(per_page=200, n_max=None)
    # Тут не разобрался как выбирать страницы для просмотра, т.е. оно идёт от начала до конца
    # Ходить чётко по страницам можно через api_url:
    # url = f'https://api.openalex.org/works?filter=institutions.id:{nsu_id}&per_page=200&page=2'
    # t = requests.get(url).json()
    # t: meta, results, в results list из 200 словарей-работ
    
    work_count = 0
    count = 1
    for page in pager:
        # conn = sqlite3.connect("D:\\opd/articles.db")
        # cursor = conn.cursor()
        print(count)
        count += 1
        # continue
        for work in page:
            
            # w = Works()['W2741809807']
            
            id_ = work['id'].split('/')[-1]
            
            """ Проверяем, есть ли эта статья в БД """
            cursor.execute('''
                            SELECT
                            EXISTS
                            (select id
                            from articles
                            where id == (?))
                            ''', (id_,))
            already_exists = cursor.fetchall()[0][0]
            if already_exists:
                # print(count)
                # count += 1
                continue
            
            # print('HERE!')
            
            column_names, content = work_parsing(work)
            number_of_columns = len(content)
            question_string = '(' + '?, '*number_of_columns
            question_string = question_string[:-2] + ')'
            
            
            cursor.execute(f"""
                            insert into articles 
                            {column_names}
                            values {question_string}""", 
                            content)
            
            work_count += 1
            if work_count % 500 == 0:
                s = f'{work_count} works downloaded'
                write_logs_by_curr_time(s)
        
        conn.commit()

def get_current_date():
    
    " Выводит строку -- текущую дату для записи логов в формате Tue_Apr_2"
    
    date = time.ctime()
    while '  ' in date:
        date = date.replace('  ', ' ')
    date = '_'.join(date.split(' ')[:3])
    return date

    
def write_logs_by_curr_time(message:str, time_data=True, path:str=None):
    
    """ Пишет логи (каждый в отдельный файл, в названии точно время старта)
        Используется в: downloading_cite_works
                        downloading_referenced_works
                        text_preprocessing
                        merging_databases
    """
    
    if path is None:
        # try:
        #     global logs_path
        #     path = logs_path
        # except:
        path = '../logs/unknown_logs.txt'
    
    s = ''
    if time_data:
        s = f'[{time.ctime()}]   '
    s += message + '\n'
    with open(path, 'a') as f:
        f.write(s)
    print(s)


def write_logs_by_curr_date(message:str, path:str=None):
    
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


def abstract_by_inv_index(inv_index):
    " Выдаёт строку -- абстракт по abstract_inverted_index "
    
    if inv_index is not None:
        if type(inv_index) is str:
            inv_index = json.loads(inv_index)
        l_inv = [(w, p) for w, pos in inv_index.items() for p in pos]
        return " ".join(map(lambda x: x[0], sorted(l_inv, key=lambda x: x[1])))


def inv_index_by_abstract(text):
    " Выдаёт словарь -- inverted_index по тексту (list of words) "
    if type(text) is str:
        if len(text) == 0:
            return None
        if text[0] == '[':
            text = json.loads(text)
        else:
            text = text.split(' ')
    elif type(text) is not list:
        print(f'Unsupported type of text: {text}')
        return
    inv_index = {}
    for word in text:
        if word not in inv_index:
            inv_index[word] = [i for i in range(len(text)) if text[i] == word]
    return inv_index


def sql_execute(request:str, args:tuple=(), size:int=None, path:str=None):
    
    """ Выполняет sql-запрос request с аргументами args,
        выдаёт первые size результатов (или все, если size is None), если есть
        в случае ошибки выдаёт и её
        формат выхода - словарь {'results': ..., 'time': ..., 'error'(если произошла): ...}
    """
    
    if path is None:
        try:
            global db_path
            path = db_path
        except:
            path ='../../main_db/articles.db'
    try:
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
    except:
        print(f"Couldn't connect to database: {path}")
    
    response = {'results': None, 'time': None}
    t_start = time.time()
    try:
        cursor.execute(request, args)
        if size is None:
            results = cursor.fetchall()
        else:
            results = cursor.fetchmany(size)
        response['results'] = results
    except:
        response['error'] = traceback.format_exc()
        try:
            conn.commit()
            conn.close()
        except:
            pass

    t = time.time() - t_start
    response['time'] = t
    try:
        conn.commit()
        conn.close()
    except:
        pass
    
    return response
    
 
def req_clear(request:str, one_line=True):
    
    " Переводит строку sql-запроса в простой вид (убирает лишние ' ' и '\n') "
    
    s = request
    s = s.split(' ')
    while '' in s:
        s.remove('')
    s = ' '.join(s)
    if one_line:
        s = ' '.join(s.split('\n')).split(' ')
        while '' in s:
            s.remove('')
        s = ' '.join(s)
    return s



if __name__ == '__main__':
    
    # download_sob_articles()
    
    pass





try:
    conn.commit()
    conn.close()
except:
    pass