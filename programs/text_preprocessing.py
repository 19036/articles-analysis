import sqlite3
import sys, os, time
import json  # для конвертации словарь <--> строка: json.dumps(dict), json.loads(str)
# import nltk
import traceback
from nltk.corpus import stopwords
from threading import Thread, Lock
# import textacy
import spacy
from textacy.preprocessing import normalize, remove, replace, make_pipeline
from helpfun import write_logs_by_curr_time
from helpfun import inv_index_by_abstract, abstract_by_inv_index, sql_execute

load_model = spacy.load('en_core_web_sm', disable = ['parser','ner'])

db_path = '../databases/math_articles.db'
logs_path = ('../logs/preprocessing/' + time.ctime().replace(' ', '___') + '_abs_cleaning.txt').replace(':', '-')
number_of_threads = 10
lock = Lock()   
try:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
except:
    print(f"Couldn't connect to database: {db_path}")


def write_logs(message:str, time_data=True):
    write_logs_by_curr_time(message, time_data, logs_path)

# def write_logs(message:str, time_data=True):
    
#     global logs_path
#     s = ''
#     if time_data:
#         s = f'[{time.ctime()}]   '
#     s += message + '\n'
#     with open(logs_path, 'a') as f:
#         f.write(s)
#     print(s)


# def invert_abstract(inv_index):
#     " Выдаёт строку -- абстракт по abstract_inverted_index "
#     if inv_index is not None:
#         l_inv = [(w, p) for w, pos in inv_index.items() for p in pos]
#         return " ".join(map(lambda x: x[0], sorted(l_inv, key=lambda x: x[1])))

# def inverted_index(text):
#     " Выдаёт словарь -- inverted_index по тексту (list of words) "
#     if type(text) == str:
#         text = text.split(' ')
#     elif type(text) != list:
#         write_logs(f'Unsupported type of text: {text}')
#         return
#     inv_index = {}
#     for word in text:
#         if word not in inv_index:
#             inv_index[word] = [i for i in range(len(text)) if text[i] == word]
#     return inv_index


def preprocessing(text:str):
    
    if text is None:
        return
    
    reproc = make_pipeline(
            load_model,
            lambda x: ' '.join(token.lemma_ for token in x),
            normalize.unicode,
            remove.accents,
            remove.brackets,
            remove.punctuation,
            replace.numbers,
            remove.brackets,
            normalize.whitespace,
            lambda x: x.lower(),
            lambda x: ' '.join([word for word in x.split(' ') if word not in stopwords.words('english')])
            )
    
    return reproc(text)

def clean_single_abstract(abstract:str):
    
    try:
        if abstract is None or abstract == 'null':
            return
        reproc = make_pipeline(
                json.loads,
                abstract_by_inv_index,
                preprocessing,
                inv_index_by_abstract,
                json.dumps
                )
        return reproc(abstract)
    except:
        s = f'Something wrong with cleaning {abstract}. Error\n:'
        s += traceback.format_exc() + '\n'
        write_logs(s)
        return

count = 0

def clean_abstracts_local(raw_data):
    
    # print(len(raw_data))
    res = list(map(lambda x: clean_single_abstract(x[1]), raw_data))
    # print(type(res[0]))
    # return
    global count
    for i in range(len(raw_data)):
        if res[i] != None:
            with lock:
                count += 1
        raw_data[i][2] = res[i]
        
    

def clean_abstracts_global(level=1):
    
    write_logs(f'\nStart cleaning abstracts of level_{level} articles\n', 0)
    
    step = 1000 # Раз в step шагов производится запись в логи
    
    global number_of_threads
    global count
    count_global = 0
    count_global_succ = 0
    
    t_global = 0
    t_start_local = time.time()
    
    request = '''
                    SELECT id, abstract
                    FROM articles
                    WHERE level = ?
                    AND abstract IS NOT NULL
                    AND abstract != ?
                    AND cleaned_abstract IS NULL
                    '''
    args = (level, 'null')
    
    response = sql_execute(request, args, size=step)
    # cursor.execute('''
    #                SELECT id, abstract
    #                FROM articles
    #                WHERE level = ?
    #                AND abstract IS NOT NULL
    #                AND abstract != ?
    #                AND cleaned_abstract IS NULL
    #                ''', (level, 'null'))
    # raw_data = cursor.fetchall()
    # print(len(raw_data))
    # return
    # raw_data = cursor.fetchmany(step)
    if 'error' in response:
        s = f'request: {request}\nargs: {args}\n'
        s += f'\n{response["error"]}'
        write_logs(s)
        return
    
    raw_data = response['results']
    raw_data = [[id_, abstract, None] for id_, abstract in raw_data]
    while raw_data is not None and raw_data != []:
        
        threads = []
        local_step = round(step/number_of_threads)
        raw_data_groups = [raw_data[local_step*i: local_step*(i+1)] for i in range(number_of_threads)]
        
        
        for i in range(number_of_threads):
            thread = Thread(target=clean_abstracts_local, args=(raw_data_groups[i],), name=f"thread_{i+1}")
            thread.start()
            threads.append(thread)
            
        for thread in threads:
            thread.join()
        
        for res in raw_data:
            id_ = res[0]
            abstract = res[2]
            try:
                cursor.execute('''
                               UPDATE articles
                               SET cleaned_abstract = ?
                               WHERE id = ?
                               ''', (abstract, id_))
                conn.commit()
            except:
                s = f'Something went wrong with inserting {(id_, abstract)}. Error:\n' 
                s += traceback.format_exc() + '\n'
                write_logs(s)
                continue
        
        t = round(time.time() - t_start_local, 1)
        t_start_local = time.time()
        t_global += t
        speed = round(count/t, 1)
        write_logs(f"{count} abstracts inserted in {t} sec, overall it's {speed} abstracts/sec")
        count_global += len(raw_data)
        count_global_succ += count
        count = 0
        
        request = '''
                        SELECT id, abstract
                        FROM articles
                        WHERE level = ?
                        AND abstract IS NOT NULL
                        AND abstract != ?
                        AND cleaned_abstract IS NULL
                        '''
        args = (level, 'null')
        
        response = sql_execute(request, args, size=step)
        
        # cursor.execute('''
        #                SELECT id, abstract
        #                FROM articles
        #                WHERE level = ?
        #                AND abstract IS NOT NULL
        #                AND abstract != ?
        #                AND cleaned_abstract IS NULL
        #                ''', (level, 'null'))
        # raw_data = cursor.fetchmany(step)
        raw_data = response['results']
        raw_data = [[id_, abstract, None] for id_, abstract in raw_data]
    
    if count_global == 0:
        write_logs(f'No raw data, all abstracts with level = {level} cleaned')
        return
    
    percent = round(count_global_succ/count_global*100, 1)
    write_logs(f'Finally {percent}% ({count_global_succ} from {count_global}) abstracts inserted in {t_global} sec')
            
        
    
    
    



if __name__ == '__main__':
    
    clean_abstracts_global(2)
    
    pass


try:
    conn.commit()
    conn.close()
except:
    pass