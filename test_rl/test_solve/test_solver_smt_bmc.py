
import json
import os
import re

import time

from z3 import *
from z3.z3 import parse_smt2_string, Solver

from test_rl.test_script.utils import solve_and_measure_time, model_to_dict, load_dictionary


start = time.time()


def test_group():
    info_name = 'info_dict_bmc.json'
    if not os.path.exists(info_name):
        # 文件不存在时，创建文件
        info_dict = {}
        with open(info_name, 'w') as file:
            json.dump(info_dict, file, indent=4)
        print(f'文件{info_name} 已创建。')
    else:
        info_dict = load_dictionary(info_name)
        print(f'文件已存在。')
    test_path = []
    directory = '/home/lz/Downloads/BMC'
    test_path.append(directory)
    # directory = '/home/lz/Downloads/incremental_Hierarchy/incremental'
    # test_path.append(directory)


    # 遍历目录
    pattern = re.compile(r'\(set-info :status (\w+)\)')
    for directory in test_path:
        path = []
        for dirpath, dirnames, filenames in os.walk(directory):
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

                    # match = pattern.search(smtlib_str)
                    # if match:
                    #     status = match.group(1)
                    #     print(f"The status is: {status}")
                    # else:
                    #     print("No status found in the file.")
                    #     return None
                    if file_path not in info_dict.keys():

                        print(type(smtlib_str))
                        try:
                            assertions = parse_smt2_string(smtlib_str)
                            solver = Solver()
                            for a in assertions:
                                solver.add(a)
                            timeout = 1200000
                            # timeout = 10

                            #先取消求解
                            result, model, time_taken = solve_and_measure_time(solver, timeout)
                        except Exception as e:
                            print(e)
                            break
                        if file_path not in info_dict.keys():
                            solve_list = []
                            solve_list.append(result)
                            solve_list.append(time_taken)
                            solve_list.append(timeout)
                            if model:
                                solve_list.append(model_to_dict(model))
                            else:
                                solve_list.append(None)
                            info_dict[file_path] = solve_list
                            with open(info_name, 'w') as file:
                                json.dump(info_dict, file, indent=4)



if __name__ == '__main__':
    test_group()
