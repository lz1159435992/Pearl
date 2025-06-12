import json
import os
from multiprocessing import Pool
from z3 import *
from z3.z3 import parse_smt2_string, Solver
from test_rl.test_script.utils import solve_and_measure_time, model_to_dict, load_dictionary
import re
def process_file(file_path, info_dict, info_name, smtlib_str):
    print(file_path)
    process_dict = {}
    assertions = parse_smt2_string(smtlib_str)
    solver = Solver()
    for a in assertions:
        solver.add(a)
    timeout = 1200000
    # timeout = 1200
    result, model, time_taken = solve_and_measure_time(solver, timeout)
    print(result, time_taken)
    result_list = [result, time_taken, timeout]
    if model:
        result_list.append(model_to_dict(model))
    else:
        result_list.append('No model')
    # print(result_list[-1])
    process_dict[file_path] = result_list
    print(process_dict)
    return process_dict

def test_group():
    tasks = []

    info_name = 'info_dict_smt_comp_QF_NIA.txt'
    if not os.path.exists(info_name):
        # 文件不存在时，创建文件
        info_dict = {}
        with open(info_name, 'w') as file:
            json.dump(info_dict, file, indent=4)
        print(f'文件{info_name} 已创建。')
    else:
        info_dict = load_dictionary(info_name)
        print(f'文件已存在。')
    NIA_dict = load_dictionary('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/NIA/NIA.json')
    test_path = []
    directory = '/home/lz/Downloads/non-incremental_Hierarchy/non-incremental'
    test_path.append(directory)
    # directory = '/home/lz/Downloads/incremental_Hierarchy/incremental'
    # test_path.append(directory)
    search_list = [
        'QF_IDL',
        # 'QF_LIA',
        # 'QF_LRA',
        # 'QF_NIA',
        # 'QF_NRA',
        # 'QF_RDL',
        # 'QF_UFIDL',
        # 'QF_UFLIA',
        # 'QF_UFLRA',
        # 'QF_UFNRA',
        # 'UFLRA',
        # 'UFNIA',
    ]
    logic_systems = [
        # "QF_BOOL",
        "QF_IDL", "QF_LIA", "QF_LRA", "QF_RDL",
        "QF_UF", "QF_UFIDL",
        # "QF_UFLIA", "QF_UFLRA", "QF_UFLIRA",
        # "BOOL",
        "LRA", "LIA",
        # "UFLIRA", "UFLRA",
        "QF_BV",
        "QF_UFBV",
        "QF_SLIA",
        "QF_BV", "QF_UFBV",
        # "QF_ABV", "QF_AUFBV", "QF_AUFLIA", "QF_ALIA", "QF_AX",
        # "QF_AUFBVLIRA",
        "QF_NRA", "QF_NIA",
        # "UFBV", "BV"
    ]
    # 遍历目录
    pattern = re.compile(r'\(set-info :status (\w+)\)')
    count = 0
    for directory in test_path:
        path = []
        for search in search_list:
            for dirpath, dirnames, filenames in os.walk(os.path.join(directory, search)):
                for filename in filenames:
                    # 构造完整的文件路径
                    file_path = os.path.join(dirpath, filename)
                    print(file_path)  # 或者进行其他操作
                    if 'starexec_description.txt' in file_path or '52759_b3ecd2335fd16ec2eee2_9_UFDTBV' in file_path or 'sll-optional-1.i_1' in file_path:
                        print('NOTHING ')
                    else:
                        with open(file_path, 'r') as file:
                            # 璇诲彇鏂囦欢鎵€鏈夊唴瀹瑰埌涓€涓瓧绗︿覆
                            smtlib_str = file.read()

                        match = pattern.search(smtlib_str)
                        if match:
                            status = match.group(1)
                            print(f"The status is: {status}")
                        else:
                            print("No status found in the file.")
                            return None
                        if file_path not in info_dict.keys() and status != 'unsat':
                            count += 1
                            if 'QF_NIA/20170427-VeryMax/ITS/16367539/From_T2__fun9.t2__p1455_edge_closing_0.smt2' in file_path:
                                print(count)
                                final_count = count
                            if file_path not in NIA_dict.keys():
                                # info_dict[file_path] = []
                                tasks.append((file_path, info_dict, info_name, smtlib_str))
    print(len(tasks))
    # print(final_count)
    with Pool() as pool:
        results = pool.starmap(process_file, tasks)

    # Combine all results into a single dictionary
    for result in results:
        info_dict.update(result)

    with open(info_name, 'w') as file:
        json.dump(info_dict, file, indent=4)

if __name__ == '__main__':
    test_group()