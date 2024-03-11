import json
import os
import sqlite3
import time
from z3 import *
value_dict = {}
result_dict = {}
value_db_path = 'value_dictionary_nju.db'
value_table_name = 'value_dictionary_nju'
result_db_path = 'result_dictionary_nju.db'
result_table_name = 'result_dictionary_nju'

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
def model_to_dict(model):
    result = {}
    for var in model:
        result[str(var)] = str(model[var])
    return result
def solve(filepath,timeout):
    conn = init_result_db(result_db_path,result_table_name)
    cursor = conn.cursor()

    cursor.execute('SELECT value FROM ' + result_table_name + ' WHERE key=?', (filepath,))
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
        smtlib_str = dict_obj['smt_script']
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

        # result_dict[filepath] = result_list
        if model:
            result_list.append(model_to_dict(model))
            conn_value = init_value_db(value_db_path, value_table_name)
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
                query_and_update(conn_value, str(model[d]), value_table_name)
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
            conn_value.close()
        else:
            result_list.append(model)
        #store dict_info
        result_insert_or_update(conn, filepath, result_list, result_table_name)
        conn.close()
        print(f"result: {result}, time: {time_taken:.2f} value: {model},filepath:{filepath}")
        #skip this file
    elif '/home/yy/Downloads/smt/gnu_angr.tar.gz/single_test/sort/sort32539' in file_path:
        print('hint')
    elif '/home/nju/Downloads/smt/smt-comp/QF_BV/Sage2_bench_17343.smt2' in file_path:
        print('hint')
if __name__ == '__main__':
    test_path = []
    # directory = '/home/yy/Downloads/smt/buzybox_angr.tar.gz/single_test'
    # test_path.append(directory)
    # directory = '/home/yy/Downloads/smt/gnu_angr.tar.gz/single_test'
    # test_path.append(directory)
    # directory = '/home/yy/Downloads/smt/gnu_KLEE/klee_bk/single_test'
    # test_path.append(directory)
    directory = '/home/nju/Downloads/smt/smt-comp/QF_BV'
    test_path.append(directory)


    # 遍历目录
    for directory in test_path:
        path = []
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                # 构造完整的文件路径
                file_path = os.path.join(dirpath, filename)
                print(file_path)  # 或者进行其他操作
                if '/home/yy/Downloads/smt/gnu_angr.tar.gz/single_test/sort/sort32539' in file_path:
                    print('hint')
                solve(file_path, 86400000)

                # try:
                #     with open(file_path, 'r') as file:
                #         # 读取文件所有内容到一个字符串
                #         smtlib_str = file.read()
                # # 解析字符串
                #     try:
                #         # 将JSON字符串转换为字典
                #         dict_obj = json.loads(smtlib_str)
                #         # print("转换后的字典：", dict_obj)
                #     except json.JSONDecodeError as e:
                #         print("解析错误：", e)

                    # 不进行判断，直接设置1天的求解时间

                    # if 'time' in dict_obj.keys() and 'solving_time_dic' in dict_obj.keys():
                    #     if float(dict_obj['time']) > 300 or float(dict_obj['solving_time_dic']['z3'][0]) > 300:
                    #         path.append(file_path)
                    #         solve(file_path,86400000)
                    #     else:
                    #         solve(file_path, 300000)
                    # elif 'time' in dict_obj.keys():
                    #     if float(dict_obj['time']) > 300:
                    #         path.append(file_path)
                    #         solve(file_path,86400000)
                    #     else:
                    #         solve(file_path, 300000)
                    # elif 'solving_time_dic' in dict_obj.keys():
                    #     if float(dict_obj['solving_time_dic']) > 300:
                    #         path.append(file_path)
                    #         solve(file_path,86400000)
                    #     else:
                    #         solve(file_path, 300000)
                    # else:
                    #     solve(file_path, 300000)
                    # print(smtlib_str)
                # except e:
                #     print('导入文件错误')
                # with open('time.txt', 'w') as file:
                #     for s in path:
                #         # 写入字符串并添加换行符
                #         file.write(s + '\n')



# print('*********************************************************************************')
# file_path = 'time.txt'
# # 使用with语句打开文件以确保正确关闭
# with open(file_path, 'w') as file:
#     for s in path:
#         # 写入字符串并添加换行符
#         file.write(s + '\n')
#
# with open('value_dict.txt', 'w') as file:
#     # 使用json.dump()将字典保存到文件
#     json.dump(value_dict, file, indent=4)
# with open('result_dict.txt', 'w') as file:
#     # 使用json.dump()将字典保存到文件
#     json.dump(result_dict, file, indent=4)
# if __name__ == '__main__':
#     solve('/home/yy/Downloads/smt/buzybox_angr.tar.gz/single_test/gnuzip/gunzip1159114',30000)




