
import json
import os

import time

from z3 import *
from z3.z3 import parse_smt2_string, Solver

from test_rl.test_script.utils import  solve_and_measure_time, model_to_dict, load_dictionary


os.environ['ALL_PROXY'] = ''
os.environ['all_proxy'] = ''

def test_group():
    info_name = 'info_dict.txt'
    if not os.path.exists(info_name):
        # 文件不存在时，创建文件
        info_dict = {}
        with open(info_name, 'w') as file:
            json.dump(info_dict, file, indent=4)
        print(f'文件{info_name} 已创建。')
    else:
        info_dict = load_dictionary(info_name)
        print(f'文件已存在。')
    with open('/home/lz/PycharmProjects/Pearl/test_rl/result_dict.txt', 'r') as file:
        result_dict = json.load(file)

    for key, value in result_dict.items():
        if '/home/yy/Downloads/' in key:
            file_path = key.replace('/home/yy/Downloads/', '/home/lz/baidudisk/')
        elif '/home/nju/Downloads/' in key:
            file_path = key.replace('/home/nju/Downloads/', '/home/lz/baidudisk/')

        with open(file_path, 'r') as file:
            # 读取文件所有内容到一个字符串
            smtlib_str = file.read()

        dict_obj = json.loads(smtlib_str)
        if 'smt-comp' in file_path:
            smtlib_str = dict_obj['smt_script']
        else:
            smtlib_str = dict_obj['script']
            python_list = json.loads(value)
        if file_path not in info_dict.keys() and python_list[0] != 'unsat':
            assertions = parse_smt2_string(smtlib_str)
            solver = Solver()
            for a in assertions:
                solver.add(a)
            # timeout = 999999999
            timeout = 1200000

            #先取消求解
            result, model, time_taken = solve_and_measure_time(solver, timeout)

            print(result, time_taken)
            # if result == 'unsat' or time_taken < 500:
            #     continue
            result_list = [result, time_taken, timeout]

            if model:
                result_list.append(model_to_dict(model))
            print(result_list[-1])

            info_dict[file_path] = result_list
            with open(info_name, 'w') as file:
                json.dump(info_dict, file, indent=4)
        elif file_path not in info_dict.keys() and python_list[0] == 'unsat':
            info_dict[file_path] = python_list
            with open(info_name, 'w') as file:
                json.dump(info_dict, file, indent=4)


if __name__ == '__main__':
    test_group()