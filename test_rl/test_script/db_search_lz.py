import sqlite3
import json
from z3 import *
from search_test import solve_and_measure_time
from utils import model_to_dict
from utils import *
import ast
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
def some_method():
    if os.path.exists('../result_dict_2.txt'):
        result_dict_2 = load_dictionary('../result_dict_2.txt')
    else:
        result_dict_2 = {}
    result_dict = load_dictionary('../result_dict.txt')
    timeout = 86400000
    for key, value in result_dict.items():
        if '/home/yy/Downloads/' in key:
            file_path = key.replace('/home/yy/Downloads/', '/home/lz/baidudisk/')
        elif '/home/nju/Downloads/' in key:
            file_path = key.replace('/home/nju/Downloads/', '/home/lz/baidudisk/')
        else:
            file_path = key
        value = ast.literal_eval(value)
        if key not in result_dict_2.keys():
            if value[0] != "unsat":
                with open(file_path, 'r') as file:
                    # 读取文件所有内容到一个字符串
                    smtlib_str = file.read()
                # 解析字符串
                try:
                    # 将JSON字符串转换为字典
                    dict_obj = json.loads(smtlib_str)
                    # print("转换后的字典：", dict_obj)
                except json.JSONDecodeError as e:
                    print("解析错误：", e)
                #
                if 'smt-comp' in file_path:
                    smtlib_str = dict_obj['smt_script']
                else:
                    smtlib_str = dict_obj['script']

                    assertions = parse_smt2_string(smtlib_str)
                    solver = Solver()
                    for a in assertions:
                        solver.add(a)
                    result, model, time_taken = solve_and_measure_time(solver, timeout)
                    print(key)
                    print(result)
                    print(model)
                    print(time_taken)
                    result_list = [result, time_taken, timeout]
                    # if result == sat:
                    #     result = 'sat'
                    # elif result == unknown:
                    #     result = 'unknown'
                    # else:
                    #     result = 'unsat'
                    if model:
                        if len(model_to_dict(model))>0:
                            result_list.append(model_to_dict(model))
                    # else:
                    #     result_list.append('无解')

                    result_dict_2[key] = result_list
            else:
                result_dict_2[key] = value
            with open('../result_dict_2.txt', 'w') as file:
                # 使用json.dump()将字典保存到文件
                json.dump(result_dict_2, file, indent=4)


if __name__ == '__main__':
    some_method()
    # db_path = 'value_dictionary.db'
    # table_name = 'value_dictionary'
    # value_dict = fetch_data_as_dict(db_path, table_name)
    # # print(value_dict)
    # count = 0
    # print(len(value_dict))
    # for k,v in value_dict.items():
    #     if int(v) > 1:
    #         # print(k)
    #         count += 1
    # print(count)

    # db_path = '/home/lz/baidudisk/3.13_db/result_dictionary.db'
    # table_name = 'result_dictionary'
    # value_dict = fetch_data_as_dict(db_path, table_name)
    # # print(value_dict)
    # count = 0
    # print(len(value_dict))
    # for k,v in value_dict.items():
    #     if 'ginstall307943' in k:
    #     # print(k,v)
    #         print(k,v)
    #         # count += 1
    # print(count)
# conn = sqlite3.connect(db_path)

# 示例数据
# cursor = conn.cursor()
# cursor.execute('SELECT * FROM result_dictionary')
# rows = cursor.fetchall()
# for i in rows:
#     list1 = json.loads(i[1])
#     if list1[0] == "unknown":
#         print(i)
# conn.close()