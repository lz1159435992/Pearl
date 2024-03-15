import sqlite3
import json
from z3 import *
from search_test import solve_and_measure_time


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
# if __name__ == '__main__':
#     result_dict_2 = {}
#     timeout = 86400000
#     db_path = 'result_dictionary.db'
#     table_name = 'result_dictionary'
#     result_dict = fetch_data_as_dict(db_path, table_name)
#     for key, value in result_dict.items():
#         list1 = json.loads(value)
#         if list1[0] == "unknown":
#             if list1[1] > 100:
#                 with open(key.replace('/home/yy/Downloads/','/home/lz/baidudisk/'), 'r') as file:
#                     # 璇诲彇鏂囦欢鎵€鏈夊唴瀹瑰埌涓€涓瓧绗︿覆
#                     smtlib_str = file.read()
#                 try:
#                     # 灏咼SON瀛楃涓茶浆鎹负瀛楀吀
#                     dict_obj = json.loads(smtlib_str)
#                     # print("杞崲鍚庣殑瀛楀吀锛?, dict_obj)
#                 except json.JSONDecodeError as e:
#                     print('failed', e)
#                 #
#                 smtlib_str = dict_obj['script']
#                 # print(smtlib_str)
#                 assertions = parse_smt2_string(smtlib_str)
#                 solver = Solver()
#                 for a in assertions:
#                     solver.add(a)
#                 result, model, time_taken = solve_and_measure_time(solver, timeout)
#                 print(key)
#                 print(result)
#                 print(model)
#                 print(time_taken)
#                 result_list = []
#                 # if result == sat:
#                 #     result = 'sat'
#                 # elif result == unknown:
#                 #     result = 'unknown'
#                 # else:
#                 #     result = 'unsat'
#                 result_list.append(result)
#                 result_list.append(time_taken)
#                 result_list.append(timeout)
#                 result_list.append(model)
#                 result_dict_2[key] = result_list
#                 with open('result_dict.txt', 'w') as file:
#                     # 使用json.dump()将字典保存到文件
#                     json.dump(result_dict, file, indent=4)
def search_script_2():
    db_path = 'result_dictionary.db'
    table_name = 'result_dictionary'
    result_dict = fetch_data_as_dict(db_path, table_name)
    for key, value in result_dict.items():
        list1 = json.loads(value)
        if list1[0] == "sat":
            if list1[1] > 50:
                print(key,value)
if __name__ == '__main__':
    search_script_2()

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