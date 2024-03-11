import sqlite3
import json
from z3 import *
from search_test import solve_and_measure_time

def update_txt_with_current_time(file_path,key,solve_time):
    with open(file_path, "a") as file:
        file.write(f"file:{key}\n" + f"solve_time:{solve_time}\n")
def fetch_data_as_dict(db_path, table_name):
    # 连接到SQLite数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 查询表中的所有键值对
    query = f"SELECT key, value FROM {table_name}"
    cursor.execute(query)
    rows = cursor.fetchall()

    # 关闭数据库连接
    conn.close()

    # 将查询结果转换为字典
    result_dict = {row[0]: row[1] for row in rows}
    return result_dict
def solve_(file_path):
    timeout = 300000
    # file_path = '/home/nju/Downloads/smt/buzybox_angr.tar.gz/single_test/udhcpc/udhcpc6668814'
    # with open(key.replace('/home/yy/Downloads/', '/home/lz/baidudisk/'), 'r') as file:
    with open(file_path, 'r') as file:
        smtlib_str = file.read()
    try:
        dict_obj = json.loads(smtlib_str)
    except json.JSONDecodeError as e:
        print('failed', e)
    smtlib_str = dict_obj['script']
    # print(smtlib_str)
    assertions = parse_smt2_string(smtlib_str)
    solver = Solver()
    for a in assertions:
        solver.add(a)
    result, model, time_taken = solve_and_measure_time(solver, timeout)
    update_txt_with_current_time('solve.txt',file_path,time_taken)
    print(result, model, time_taken)
def search_script_2(db_path, table_name):
    # db_path = 'result_dictionary.db'
    # table_name = 'result_dictionary'
    result_dict = fetch_data_as_dict(db_path, table_name)
    for key, value in result_dict.items():
        list1 = json.loads(value)
        if list1[0] == "sat":
            if list1[1] > 20:
                print(key,value)
                # solve_(key.replace('/home/yy/Downloads/','/home/nju/Downloads/'))
def search_script_watch(db_path,table_name):
    # db_path = 'result_dictionary.db'
    # table_name = 'result_dictionary'
    result_dict = fetch_data_as_dict(db_path, table_name)
    for key, value in result_dict.items():
        print(key, value)
if __name__ == '__main__':
    search_script_2('result_dictionary_nju.db', 'result_dictionary_nju')

    # search_script_watch('result_dictionary_nju.db', 'result_dictionary_nju')
    # search_script_watch('value_dictionary_nju.db', 'value_dictionary_nju')
    # result_list = []
    # result_list.append(result)
    # result_list.append(time_taken)
    # result_list.append(timeout)
    # result_list.append(model)
    # result_dict_2[key] = result_list
    # with open('result_dict.txt', 'w') as file:
    #     # 使用json.dump()将字典保存到文件
    #     json.dump(result_dict, file, indent=4)

