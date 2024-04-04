import sqlite3
import sys, os, time
from helpfun import write_logs_by_curr_time

# db_path_from = '../databases/local_db/articles.db'
db_path_from = '../databases/articles.db'
table_name_from = 'articles'
# db_path_to = '../databases/articles.db'
# db_path_to = '../databases/sob_articles.db'
db_path_to = '../databases/nsu_articles.db'
table_name_to = 'articles'

logs_path = ('../logs/merge_tables/' + time.ctime().replace(' ', '___') + '_merge.txt').replace(':', '-')

def write_logs(message:str, time_data=True):
    write_logs_by_curr_time(message, time_data, logs_path)

try:
    conn_from = sqlite3.connect(db_path_from)
    cursor_from = conn_from.cursor()
except:
    print(f"Couldn't connect to database: {db_path_from}")
    
try:
    conn_to = sqlite3.connect(db_path_to)
    cursor_to = conn_to.cursor()
except:
    print(f"Couldn't connect to database: {db_path_to}")
    

""" Создание таблицы """
# cursor_to.execute("""
# CREATE TABLE IF NOT EXISTS test (
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
# cursor_from.execute('''
#                     ALTER TABLE articles
#                     ADD cleaned_abstract TEXT
#                     ''')


def merge_tables():
    
    global table_name_from
    global table_name_to
    
    write_logs(f'Start merging tables:\nFrom {db_path_from}, {table_name_from} to {db_path_to}, {table_name_to}\n', 0)
    
    t_start = time.time()
    
    cursor_from.execute(f'''
                        SELECT *
                        FROM {table_name_from}
                        WHERE level = 1
                        ''')
    
    results = cursor_from.fetchall()
    
    t_got_values = round(time.time()-t_start, 1)
    write_logs(f'Got values in {t_got_values} sec')
    t_start = time.time()
    
    number_of_columns = len(results[0])
    q_str = '(' + '?, '*(number_of_columns-1) + '?)'
    count = 0
    already_in_db_ids = []
    
    for res in results:
        
        """ Проверяем, есть ли эта статья в БД """
        cursor_to.execute(f'''
                        SELECT
                        EXISTS
                        (select id
                        from {table_name_to}
                        where id == (?))
                        ''', (res[0],))
        already_exists = cursor_to.fetchall()[0][0]
        if already_exists:
            already_in_db_ids.append(res[0])
            count += 1
            continue
        
        cursor_to.execute(f"""
                           INSERT INTO {table_name_to}
                           VALUES
                           {q_str}
                           """, res)
    
    t_merge =  round(time.time()-t_start, 1)
    write_logs(f'Merged in {t_merge} sec')
    if count != 0:
        s = f'{count} from {len(results)} articles already exist, their ids:\n'
        s += ' '.join(already_in_db_ids)
        write_logs(s)



if __name__ == '__main__':
    
    merge_tables()
    
    
    pass



try:
    conn_from.commit()
    conn_from.close()
except:
    pass

try:
    conn_to.commit()
    conn_to.close()
except:
    pass