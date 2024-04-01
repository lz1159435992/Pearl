import json
import os
import sqlite3
import time
from z3 import *
value_dict = {}
result_dict = {}
value_db_path = 'value_dictionary.db'
result_db_path = 'result_dictionary.db'
def init_result_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS result_dictionary (
            key TEXT UNIQUE, 
            value TEXT
        )
    ''')
    conn.commit()
    return conn

def insert_result_dict_to_db(conn, dict_path):
    cursor = conn.cursor()
    dictionary = load_dictionary(dict_path)
    for key, value in dictionary.items():
        print(value)
        value_str = json.dumps(value)
        cursor.execute('INSERT OR IGNORE INTO result_dictionary (key, value) VALUES (?, ?)', (key, value_str))
    conn.commit()
# 将键值对插入或更新数据库
def result_insert_or_update(conn, key, value_list):
    cursor = conn.cursor()
    # 序列化列表为JSON字符串
    value_str = json.dumps(value_list)
    cursor.execute('''
        INSERT INTO result_dictionary (key, value) 
        VALUES (?, ?)
        ON CONFLICT(key) 
        DO UPDATE SET value=excluded.value
    ''', (key, value_str))
    conn.commit()


# 查询并更新键值列表
def result_query_and_update(conn, key, new_values):
    cursor = conn.cursor()
    cursor.execute('SELECT value FROM result_dictionary WHERE key=?', (key,))
    result = cursor.fetchone()
    if result:
        # 反序列化JSON字符串为列表
        existing_values = json.loads(result[0])
        # 更新列表
        updated_values = existing_values + new_values
        # 重新序列化列表为JSON字符串进行更新
        result_insert_or_update(conn, key, updated_values)
    else:
        # 如果键不存在，则添加键值对
        insert_or_update(conn, key, new_values)


def load_dictionary(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


# 初始化数据库和表
def init_value_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS value_dictionary (key TEXT UNIQUE, value INTEGER)')
    conn.commit()
    return conn


# 将字典插入数据库
def insert_value_dict_to_db(conn, dictionary):
    cursor = conn.cursor()
    for key, value in dictionary.items():
        cursor.execute('INSERT OR IGNORE INTO value_dictionary (key, value) VALUES (?, ?)', (key, value))
    conn.commit()


# 查询并更新键值
def query_and_update(conn, key):
    cursor = conn.cursor()
    cursor.execute('SELECT value FROM value_dictionary WHERE key=?', (key,))
    result = cursor.fetchone()
    if result:
        # 如果键存在，则更新其值
        new_value = result[0] + 1
        cursor.execute('UPDATE value_dictionary SET value=? WHERE key=?', (new_value, key))
    else:
        # 如果键不存在，则添加键值对，这里假设新键的初始值为1
        cursor.execute('INSERT INTO value_dictionary (key, value) VALUES (?, ?)', (key, 1))
    conn.commit()

# 指定需要遍历的目录
def solve_and_measure_time(solver, timeout):
    solver.set("timeout", timeout)
    start_time = time.time()
    result = solver.check()
    # print(result)
    elapsed_time = time.time() - start_time
    if result == sat:
        return "sat", solver.model(), elapsed_time
    elif result == unknown:
        return "unknown", None, elapsed_time
    else:
        return "unsat", None, elapsed_time

def solve(filepath,timeout):
    conn = init_result_db('result_dictionary.db')
    cursor = conn.cursor()

    cursor.execute('SELECT value FROM result_dictionary WHERE key=?', (filepath,))
    result = cursor.fetchone()
    if result is None:
        # 键不存在，添加键值对
        with open(filepath, 'r') as file:
            # 璇诲彇鏂囦欢鎵€鏈夊唴瀹瑰埌涓€涓瓧绗︿覆
            smtlib_str = file.read()
        try:
            # 灏咼SON瀛楃涓茶浆鎹负瀛楀吀
            dict_obj = json.loads(smtlib_str)
            # print("杞崲鍚庣殑瀛楀吀锛?, dict_obj)
        except json.JSONDecodeError as e:
            print('failed', e)
        #
        smtlib_str = dict_obj['script']
        # print(smtlib_str)
        assertions = parse_smt2_string(smtlib_str)
        solver = Solver()
        for a in assertions:
            solver.add(a)
        result, model, time_taken = solve_and_measure_time(solver, timeout)
        result_list = []
        # if result == sat:
        #     result = 'sat'
        # elif result == unknown:
        #     result = 'unknown'
        # else:
        #     result = 'unsat'
        result_list.append(result)
        result_list.append(time_taken)
        result_list.append(timeout)
        #store dict_info
        result_insert_or_update(conn, filepath, result_list)
        # result_dict[filepath] = result_list
        if model:
            conn = init_value_db('value_dictionary.db')
            #
            # with open('value_dict.txt', 'r') as value_file:
            #     # 璇诲彇鏂囦欢鎵€鏈夊唴瀹瑰埌涓€涓瓧绗︿覆
            #     str = value_file.read()
            # try:
            #     # 灏咼SON瀛楃涓茶浆鎹负瀛楀吀
            #     value_dict = json.loads(str)
            #     # print("杞崲鍚庣殑瀛楀吀锛?, dict_obj)
            # except json.JSONDecodeError as e:
            #     print('failed', e)
            for d in model.decls():
                query_and_update(conn, str(model[d]))
            #     if str(model[d]) not in value_dict.keys():
            #         value_dict[str(model[d])] = 1
            #     else:
            #         value_dict[str(model[d])] += 1
            # with open('value_dict.txt', 'w') as file:
            #     # 使用json.dump()将字典保存到文件
            #     json.dump(value_dict, file, indent=4)
            # with open('result_dict.txt', 'w') as file:
            #     # 使用json.dump()将字典保存到文件
            #     json.dump(result_dict, file, indent=4)
        print(f"result: {result}, time: {time_taken:.2f} value: {model},filepath:{filepath}")
        #skip this file
    elif '/home/yy/Downloads/smt/gnu_angr.tar.gz/single_test/sort/sort32539' in file_path:
        print('hint')
if __name__ == '__main__':
    test_path = []
    # directory = '/home/yy/Downloads/smt/buzybox_angr.tar.gz/single_test'
    # test_path.append(directory)
    # directory = '/home/yy/Downloads/smt/gnu_angr.tar.gz/single_test'
    # test_path.append(directory)
    # directory = '/home/yy/Downloads/smt/gnu_KLEE/klee_bk/single_test'
    # test_path.append(directory)
    # directory = '/home/yy/Downloads/smt/smt-comp/QF_BV'
    # test_path.append(directory)
    directory ='/media/lz/Seagate Expansion Drive/ffmpeg/cons'
    test_path.append(directory)



    # 遍历目录
    for directory in test_path:
        path = []
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                # 构造完整的文件路径
                file_path = os.path.join(dirpath, filename)
                print(file_path)  # 或者进行其他操作
                solve(file_path, 9000000)





