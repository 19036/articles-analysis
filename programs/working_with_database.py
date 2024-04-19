import sqlite3
import json
import sys, time
import traceback
from helpfun import get_current_date
from helpfun import sql_execute, req_clear

# db_path = '../databases/local_db/articles.db'
# db_path = '../databases/articles.db'
# db_path = '../databases/sob_articles.db'
db_path = '../databases/nsu_articles.db'
# db_path = '../databases/distances/distances_nsu_level_1.db'
logs_path = f'../logs/working_with_database/{get_current_date()}.txt'
try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
except:
    print(f"Couldn't connect to database: {db_path}")

""" Создание таблицы """
# cursor.execute("""
# CREATE TABLE IF NOT EXISTS articles (
#     id TEXT PRIMARY KEY,
#     title TEXT,
#     authors TEXT,
#     abstract TEXT,
#     referenced_works TEXT,
#     cites_this_work TEXT,
#     cited_by_count INT,
#     publication_year INT,
#     keywords_oa TEXT,
#     topics_oa TEXT,
#     level INT,
#     cleaned_abstract TEXT
# )
# """)



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

def lists_of_works_change_format():
    
    request = 'SELECT id, referenced_works, cites_this_work FROM articles'
    cursor.execute(request)
    results = cursor.fetchall()
    t_start = time.time()
    count = 0
    for id_, ref_list, cite_list in results:
        r_change, c_change = 0, 0
        if ref_list is not None:
            if len(ref_list) > 0 and ref_list[0] != '[':
                ref_list = json.dumps(ref_list.split(', '))
                r_change = 1
        if cite_list is not None:
            if len(cite_list) > 0 and cite_list[0] != '[':
                cite_list = json.dumps(cite_list.split(', '))
                l_change = 1
        # print(id_)
        # print(ref_list)
        # request = f'UPDATE articles SET referenced_works = {ref_list} WHERE id = {id_}'
        if r_change:
            request = 'UPDATE articles SET referenced_works = ? WHERE id = ?'
            cursor.execute(request, (ref_list, id_))
            conn.commit()
        if l_change:
            request = 'UPDATE articles SET cites_this_work = ? WHERE id = ?'
            cursor.execute(request, (cite_list, id_))
            conn.commit()
        count += 1
        if count % 10000 == 0:
            count = 0
            t = round(time.time() - t_start, 1)
            print(f'10000 articles checked in {t} sec')
    
def delete_empty_abstracts():
    t_start = time.time()
    # request = 'DELETE FROM articles WHERE abstract IS NULL OR abstract = "null"'
    request = 'DELETE FROM articles WHERE cleaned_abstract IS NULL OR cleaned_abstract = "null"'
    sql_execute(request, path=db_path)
    return round(time.time() - t_start, 1)

def update_ref_lists(level=1):
    t_start = time.time()
    request = f'SELECT id, referenced_works FROM articles WHERE level = {level}'
    cursor.execute(request)
    results = cursor.fetchall()
    for id_, ref_list in results:
        if ref_list is None or len(ref_list) == 0:
            continue
        ref_list = json.loads(ref_list)
        new_ref_list = []
        for ref_id in ref_list:
            request = f'SELECT EXISTS (SELECT id FROM articles WHERE id = ?)'
            cursor.execute(request, (ref_id,))
            exists = cursor.fetchall()[0][0]
            if exists:
                new_ref_list.append(ref_id)
        new_ref_list = json.dumps(new_ref_list)
        request = 'UPDATE articles SET referenced_works = ? WHERE id = ?'
        cursor.execute(request, (new_ref_list, id_))
        conn.commit()
    
    return round(time.time() - t_start, 1)   

def update_cite_lists(level=1):
    t_start = time.time()
    request = f'SELECT id, cites_this_work FROM articles WHERE level = {level}'
    cursor.execute(request)
    results = cursor.fetchall()
    for id_, cite_list in results:
        if cite_list is None or len(cite_list) == 0:
            continue
        cite_list = json.loads(cite_list)
        new_cite_list = []
        for cite_id in cite_list:
            request = f'SELECT EXISTS (SELECT id FROM articles WHERE id = ?)'
            cursor.execute(request, (cite_id,))
            exists = cursor.fetchall()[0][0]
            if exists:
                new_cite_list.append(cite_id)
        new_cite_list = json.dumps(new_cite_list)
        request = f'UPDATE articles SET cites_this_work = ? WHERE id = ?'
        cursor.execute(request, (new_cite_list, id_))
        conn.commit()
    
    return round(time.time() - t_start, 1)   

def clear_database():
    
    s = f'Start clearing database: {db_path}\n'
    t = delete_empty_abstracts()
    print(f'Empty abstracts deleted in {t} sec\n')
    s += f'Empty abstracts deleted in {t} sec\n'
    t = update_ref_lists()
    print(f'Ref_lists updated in {t} sec\n')
    s += f'Ref_lists updated in {t} sec\n'
    t = update_cite_lists()
    print(f'Cite_lists updated in {t} sec\n')
    s += f'Cite_lists updated in {t} sec\n'
    write_logs(s)


def alter_req():
    
    args = ()
    
    # request = '''
    #             SELECT cited_by_count
    #             FROM articles
    #             WHERE level = ?
    #             '''
    # args = (2,)
    
    # request = '''
    #             SELECT referenced_works
    #             FROM articles
    #             WHERE level = ?
    #             '''
    # args = (2,)
    
    request = 'SELECT COUNT(*) FROM articles '
    # request += 'WHERE abstract = ? OR abstract IS NULL'
    # args = ('null',)
    # request = 'ALTER TABLE articles ADD COLUMN cleaned_abstract TEXT'
    # request = 'ALTER TABLE distances RENAME COLUMN distance TO distances'
    
    request = req_clear(request)
    
    response = sql_execute(request, args, path=db_path)
    
    global logs_path
    path = logs_path + 'requests.txt'
    s = f'database: {db_path.split("/")[-1]}'
    s += f'request: {request}\nargs: {args}\n'
    if 'error' in response:
        s += f'\n{response["error"]}'
        write_logs(s, path)
        return
    t = round(response['time'], 1)
    results = response['results']
    
    " Для COUNT(*) "
    count = results[0][0]
    s += f'count: {count}\n'
    
    " Для sum(cited_by_count) "
    # count = sum(list(map(lambda x: x[0], results)))
    # s += f'sum: {count}\n'
    
    " Для sum(len(ref_lists)) "
    # count = sum(list(map(lambda x: len(x[0]), results)))
    # s += f'sum: {count}\n'
    
    
    
    s += f'time: {t}'
    write_logs(s, path)
    

def compile_topics_oa(db_name:str='nsu', level=None, unique=1):
    
    if level is None:
        request = 'SELECT topics_oa FROM articles'
    else:
        request = f'SELECT topics_oa FROM articles WHERE level = {level}'
    db_path = f'../databases/{db_name}_articles.db'
    response = sql_execute(request, path=db_path)
    results = response['results']
    topics_dict = {'topics': {}, 'subfields': {}, 'fields': {}, 'domains': {}}
    t_start = time.time()
    for res in results:
        if res[0] is None or len(res[0]) < 4:
            continue
        res = json.loads(res[0])
        for topic in res:
            name, subfield, field, domain = topic['name'], topic['subfield'], topic['field'], topic['domain']
            
            if name in topics_dict['topics']:
                topics_dict['topics'][name] += 1
            else:
                topics_dict['topics'][name] = 1
                
            if subfield in topics_dict['subfields']:
                topics_dict['subfields'][subfield] += 1
            else:
                topics_dict['subfields'][subfield] = 1
            
            if field in topics_dict['fields']:
                topics_dict['fields'][field] += 1
            else:
                topics_dict['fields'][field] = 1
            
            if domain in topics_dict['domains']:
                topics_dict['domains'][domain] += 1
            else:
                topics_dict['domains'][domain] = 1
            
            if unique:
                break
    
    for key in topics_dict:
        topics_dict[key] = dict(sorted(topics_dict[key].items(), key=lambda elem: elem[1], reverse=True))
    topics_dict = json.dumps(topics_dict)
    if level is None:
        path = f'../some info/{db_name}_topics_dict'
    else:
        path = f'../some info/{db_name}_level_{level}_topics_dict'
    if unique:
        path += '_uniq'
    path += '.txt'
    with open(path, 'w') as f:
        f.write(topics_dict)
    
    t = round(time.time() - t_start, 1)
    print(f'Topics_dict for {db_name} articles compiled in {t} sec')
            
        
def get_topics_oa_info(db_name:str='nsu', level=None, unique=1):
    
    if level is None:
        path = f'../some info/{db_name}_topics_dict'
    else:
        path = f'../some info/{db_name}_level_{level}_topics_dict'
    if unique:
        path += '_uniq'
    path += '.txt'
    with open(path, 'r') as f:
        td = json.loads(f.read())
    
    if level is None:
        s = f'About topics_oa of {db_name} (unique:{unique}) articles:\n\n'
    else:
        s = f'About topics_oa of {db_name} level_{level} (unique:{unique}) articles:\n\n'
    s += 'Domains:\n'
    for domain in td['domains']:
        s += f"{domain}: {td['domains'][domain]}\n"
    s += f"Fields: {len(td['fields'].keys())}\n"
    for field in td['fields']:
        s += f'    {field}: {td["fields"][field]}\n'
    s += f"Subfields: {len(td['subfields'].keys())}\n"
    s += f"Topics: {len(td['topics'].keys())}\n"
    
    write_logs(s)
    

    
    
    
    

if __name__ == '__main__':
    
    # alter_req()
    
    # clear_database()
    
    # lists_of_works_change_format()
    
    pass


def archive():
    
    """ Разные куски функций, которые могут пригодиться"""
    
    
    """ Некоторые команды курсора sqlite3 """
    
    ' Добавить столбец '
    # request = 'ALTER TABLE articles ADD COLUMN cites_this_work TEXT'
    ' Удаление столбцов '
    # request = 'ALTER TABLE articles DROP COLUMN id'
    ' Переименовать столбец '
    # request = 'ALTER TABLE articles RENAME COLUMN mane TO title'
    ' Создать индекс (пока хз че это) '
    # request = 'CREATE INDEX idx_email ON users (email)'
    ' Вставить элемент (строчку) '
    # request, args = 'INSERT INTO users (name, age, email) values (?, ?, ?)', ('who', None, 'hramov@')
    ' Изменить элемент(ы) '
    # request, args = 'UPDATE users SET age = ? WHERE id = ?', (21, 4)
    ' Удалить элемент(ы) '
    # request, args = 'DELETE FROM articles WHERE id = ?', (4,)
    ' Выбор строк '
    # request = '''
    #                SELECT id, name, age 
    #                FROM users
    #                WHERE age IS NULL
    #                '''
    # results = cursor.fetchall()
    ' Число элементов '
    # request = 'SELECT COUNT(*) FROM articles WHERE abstract IS NULL'
    
    
    
    
    
    """ Старые функции по arxiv """
    
    " Сохраняет pdf - файл по данной url "
    
    # url = 'https://arxiv.org/pdf/2402.00001.pdf'
    # urllib.request.urlretrieve(url, 'testpd.pdf')
    
    
    " Читает pdf файл, можно вытащить весь текст "
    
    # reader = pypdf.PdfReader('testpd.pdf')
    # print(reader.pages[1].extract_text())
    
    
    " Проходится по статьям из тестовых 2000 (щас - по первым 20 из них), "
    " качает их как pdf, считывает текст, записывает в БД, удаляет pdf    "
    
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
    
    
    pass


try:
    conn.commit()
    conn.close()
except:
    pass


