import sqlite3
import json
import sys, time
import traceback


db_path = '../../main_db/articles.db'
date = '_'.join(time.ctime().split(' ')[:3])
logs_path = f'../logs/working_with_database/{date}_'
try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
except:
    print(f"Couldn't connect to database: {db_path}")



def write_logs(message:str, path:str=logs_path):
    
    if path[-4:] != '.txt':
        path += '_unknown.txt'
        
    s = f'[{time.ctime()}]\n'
    s += message + '\n\n\n'
    with open(path, 'a') as f:
        f.write(s)
    print(s)


def sql_execute(request:str, args:tuple, size:int=None):
    
    global db_path
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    except:
        print(f"Couldn't connect to database: {db_path}")
    
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
        


def database_changing():
    
    ' Добавить столбец '
    # request = 'ALTER TABLE articles ADD COLUMN cites_this_work TEXT'
    
    pass


def alter_req():
    
    # request = '''
    #             SELECT cited_by_count
    #             FROM articles
    #             WHERE level = ?
    #             '''
    # args = (2,)
    
    request = '''
                SELECT referenced_works
                FROM articles
                WHERE level = ?
                '''
    args = (2,)
    
    # request = 'SELECT COUNT(*) FROM articles '
    # request += 'WHERE level = ?'
    # args = (1,)
    
    request = req_clear(request)
    
    response = sql_execute(request, args)
    
    global logs_path
    path = logs_path + 'requests.txt'
    s = f'request: {request}\nargs: {args}\n'
    if 'error' in response:
        s += f'\n{response["error"]}'
        write_logs(s, path)
        return
    t = round(response['time'], 1)
    results = response['results']
    
    " Для COUNT(*) "
    # count = results[0][0]
    # s += f'count: {count}\n'
    
    " Для sum(cited_by_count) "
    # count = sum(list(map(lambda x: x[0], results)))
    # s += f'sum: {count}\n'
    
    " Для sum(len(ref_lists)) "
    count = sum(list(map(lambda x: len(x[0]), results)))
    s += f'sum: {count}\n'
    
    
    
    s += f'time: {t}'
    write_logs(s, path)
    



if __name__ == '__main__':
    
    # alter_req()
    
    
    
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


