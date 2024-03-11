import sqlite3
import json

def load_dictionary(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
# 初始化数据库和表
def init_result_db(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS '''+table_name+''' (
            key TEXT UNIQUE, 
            value TEXT
        )
    ''')
    conn.commit()
    return conn

def insert_result_dict_to_db(conn, dict_path, table_name):
    cursor = conn.cursor()
    dictionary = load_dictionary(dict_path)
    for key, value in dictionary.items():
        print(value)
        value_str = json.dumps(value)
        cursor.execute('INSERT OR IGNORE INTO ' + table_name + ' (key, value) VALUES (?, ?)', (key, value_str))
    conn.commit()
# 将键值对插入或更新数据库
def result_insert_or_update(conn, key, value_list, table_name):
    cursor = conn.cursor()
    # 序列化列表为JSON字符串
    value_str = json.dumps(value_list)
    cursor.execute('''
        INSERT INTO ''' + table_name + ''' (key, value) 
        VALUES (?, ?)
        ON CONFLICT(key) 
        DO UPDATE SET value=excluded.value
    ''', (key, value_str))
    conn.commit()


# 查询并更新键值列表
def result_query_and_update(conn, key, new_values, tabele_name):
    cursor = conn.cursor()
    cursor.execute('SELECT value FROM ' + tabele_name + ' WHERE key=?', (key,))
    result = cursor.fetchone()
    if result:
        # 反序列化JSON字符串为列表
        existing_values = json.loads(result[0])
        # 更新列表
        updated_values = existing_values + new_values
        # 重新序列化列表为JSON字符串进行更新
        insert_or_update(conn, key, updated_values)
    else:
        # 如果键不存在，则添加键值对
        insert_or_update(conn, key, new_values)

def init_value_db(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS ' + table_name + ' (key TEXT UNIQUE, value INTEGER)')
    conn.commit()
    return conn


# 将字典插入数据库
def insert_value_dict_to_db(conn, dictionary, table_name):
    cursor = conn.cursor()
    for key, value in dictionary.items():
        cursor.execute('INSERT OR IGNORE INTO ' + table_name + ' (key, value) VALUES (?, ?)', (key, value))
    conn.commit()


# 查询并更新键值
def query_and_update(conn, key, table_name):
    cursor = conn.cursor()
    cursor.execute('SELECT value FROM ' + table_name + ' WHERE key=?', (key,))
    result = cursor.fetchone()
    if result:
        # 如果键存在，则更新其值
        new_value = result[0] + 1
        cursor.execute('UPDATE ' + table_name + ' SET value=? WHERE key=?', (new_value, key))
    else:
        # 如果键不存在，则添加键值对，这里假设新键的初始值为1
        cursor.execute('INSERT INTO ' + table_name + ' (key, value) VALUES (?, ?)', (key, 1))
    conn.commit()
def init_all_db():
    db_path = 'result_dictionary_nju.db'
    table_name = 'result_dictionary_nju'
    conn = init_result_db(db_path, table_name)
    conn.close()

    db_path = 'value_dictionary_nju.db'
    table_name = 'value_dictionary_nju'

    conn = init_value_db(db_path,table_name)
    conn.close()
# 主函数
def main():
    db_path = 'result_dictionary.db'
    conn = init_result_db(db_path)

    # 示例数据
    insert_result_dict_to_db(conn, 'result_dict.txt')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM result_dictionary')
    rows = cursor.fetchall()
    print(rows)
    conn.close()

    db_path = 'value_dictionary.db'
    dict_file_path = 'value_dict.txt'
    dictionary = load_dictionary(dict_file_path)
    conn = init_value_db(db_path)
    insert_value_dict_to_db(conn, dictionary)
    conn.close()

if __name__ == '__main__':
    #main()
    init_all_db()
