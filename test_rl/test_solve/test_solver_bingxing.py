import json
import os
from multiprocessing import Pool
from z3 import *
from z3.z3 import parse_smt2_string, Solver
from test_rl.test_script.utils import solve_and_measure_time, model_to_dict, load_dictionary

def process_file(file_path, python_list, info_dict, info_name, smtlib_str):
    print(file_path)
    process_dict = {}
    if python_list[0] != 'unsat':
        assertions = parse_smt2_string(smtlib_str)
        solver = Solver()
        for a in assertions:
            solver.add(a)
        timeout = 1200000
        result, model, time_taken = solve_and_measure_time(solver, timeout)
        print(result, time_taken)
        result_list = [result, time_taken, timeout]
        if model:
            result_list.append(model_to_dict(model))
        else:
            result_list.append('No model')
        print(result_list[-1])
        process_dict[file_path] = result_list
    elif python_list[0] == 'unsat':
        process_dict[file_path] = python_list
    return process_dict

def test_group():
    info_name = 'info_dict.txt'
    if not os.path.exists(info_name):
        info_dict = {}
        with open(info_name, 'w') as file:
            json.dump(info_dict, file, indent=4)
        print(f'文件{info_name} 已创建。')
    else:
        info_dict = load_dictionary(info_name)
        print(f'文件已存在。')

    with open('/home/lz/PycharmProjects/Pearl/test_rl/result_dict.txt', 'r') as file:
        result_dict = json.load(file)

    tasks = []
    for key, value in result_dict.items():
        if '/home/yy/Downloads/' in key:
            file_path = key.replace('/home/yy/Downloads/', '/home/lz/baidudisk/')
        elif '/home/nju/Downloads/' in key:
            file_path = key.replace('/home/nju/Downloads/', '/home/lz/baidudisk/')
        else:
            file_path = key
        if file_path not in info_dict.keys():

            with open(file_path, 'r') as file:
                smtlib_str = file.read()
            dict_obj = json.loads(smtlib_str)
            if 'smt-comp' in file_path:
                smtlib_str = dict_obj['smt_script']
            else:
                smtlib_str = dict_obj['script']
                python_list = json.loads(value)
            tasks.append((file_path, python_list, info_dict, info_name, smtlib_str))

    with Pool() as pool:
        results = pool.starmap(process_file, tasks)

    # Combine all results into a single dictionary
    for result in results:
        info_dict.update(result)

    with open('info_dict_bingxing.txt', 'w') as file:
        json.dump(info_dict, file, indent=4)

if __name__ == '__main__':
    test_group()