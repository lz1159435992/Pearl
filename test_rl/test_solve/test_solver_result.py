
import json
import os

import time

from z3 import *
from z3.z3 import parse_smt2_string, Solver

from test_rl.test_script.utils import  solve_and_measure_time, model_to_dict, load_dictionary, setup_logger
from loguru import logger


os.environ['ALL_PROXY'] = ''
os.environ['all_proxy'] = ''
setup_logger()

def test_group():
    info_name = 'info_dict_gai_6_normal_1109_pre_SMTimer_llama3.1:70b_1200s_result.txt'
    if not os.path.exists(info_name):
        # 文件不存在时，创建文件
        info_dict = {}
        with open(info_name, 'w') as file:
            json.dump(info_dict, file, indent=4)
        print(f'文件{info_name} 已创建。')
    else:
        info_dict = load_dictionary(info_name)
        print(f'文件已存在。')
    with open('/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1109_pre_SMTimer_llama3.1:70b_1200s.txt', 'r') as file:
        result_dict = json.load(file)

    for key, value in result_dict.items():
        if '/home/yy/Downloads/' in key:
            file_path = key.replace('/home/yy/Downloads/', '/home/lz/baidudisk/')
        elif '/home/nju/Downloads/' in key:
            file_path = key.replace('/home/nju/Downloads/', '/home/lz/baidudisk/')
        else:
            file_path = key
        with open(file_path, 'r') as file:
            # 读取文件所有内容到一个字符串
            smtlib_str = file.read()

        dict_obj = json.loads(smtlib_str)
        if 'smt-comp' in file_path:
            smtlib_str = dict_obj['smt_script']
        else:
            smtlib_str = dict_obj['script']
            # python_list = json.loads(value)
        if file_path not in info_dict.keys():
            assertions = parse_smt2_string(smtlib_str)
            solver = Solver()
            for a in assertions:
                solver.add(a)
            # timeout = 999999999
            timeout = 1200000

            #先取消求解
            result, model, time_taken = solve_and_measure_time(solver, timeout)
            print(result_dict[key][1])
            result_dict[key][1] = time_taken
            print(result, time_taken)
            # if result == 'unsat' or time_taken < 500:
            #     continue
            # result_list = [result, time_taken, timeout]
            #
            # if model:
            #     result_list.append(model_to_dict(model))
            # print(result_list[-1])
            #
            # info_dict[file_path] = result_list
            info_dict[key] = result_dict[key]
            with open(info_name, 'w') as file:
                json.dump(info_dict, file, indent=4)
        # elif file_path not in info_dict.keys() and python_list[0] == 'unsat':
        #     info_dict[file_path] = python_list
        #     with open(info_name, 'w') as file:
        #         json.dump(info_dict, file, indent=4)
#选择修正过的求解时间大于300s的测试数据
def test_group_1():
    info_name = 'info_dict_gai_6_normal_1109_pre_SMTimer_llama3.1:70b_1200s_result.txt'
    if not os.path.exists(info_name):
        # 文件不存在时，创建文件
        info_dict = {}
        with open(info_name, 'w') as file:
            json.dump(info_dict, file, indent=4)
        print(f'文件{info_name} 已创建。')
    else:
        info_dict = load_dictionary(info_name)
        print(f'文件已存在。')
    count = 0
    succeed_count = 0
    failed_count = 0
    all_time_sat_succeed = 0
    all_time_rl_succeed = 0
    all_time_sat_failed = 0
    all_time_rl_failed = 0
    all_time_sat = 0
    all_time_rl = 0
    for k,v in info_dict.items():
       if v[1] > 300:
            print(k,v[0],v[1],v[2],v[3],v[4])
            count += 1
            all_time_sat += v[1]
            all_time_rl += v[3]
            if v[4] == 'succeed':
                succeed_count += 1
                all_time_sat_succeed += v[1]
                all_time_rl_succeed += v[3]
            elif v[4] == 'failed':
                failed_count += 1
                all_time_sat_failed += v[1]
                all_time_rl_failed += v[3]
    print(count,succeed_count,failed_count,all_time_sat,all_time_rl,all_time_sat_succeed,all_time_rl_succeed,all_time_sat_failed,all_time_rl_failed)

#选择修正过的求解时间大于300s的测试数据
def test_group_1_save():
    solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict.txt'
    solve_dict = load_dictionary(solve_name)

    info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1109_pre_SMTimer_save_docker_llama_3.1:70b_1200s.txt'
    info_dict = load_dictionary(info_name)
    for k,v in info_dict.items():
        if k in solve_dict.keys():
            info_dict[k][0] = solve_dict[k][0]
            info_dict[k][1] = solve_dict[k][1]

    # if not os.path.exists(info_name):
    #     # 文件不存在时，创建文件
    #     info_dict = {}
    #     with open(info_name, 'w') as file:
    #         json.dump(info_dict, file, indent=4)
    #     print(f'文件{info_name} 已创建。')
    # else:
    #     info_dict = load_dictionary(info_name)
    #     print(f'文件已存在。')
    count = 0
    succeed_count = 0
    failed_count = 0
    all_time_sat_succeed = 0
    all_time_rl_succeed = 0
    all_time_sat_failed = 0
    all_time_rl_failed = 0
    all_time_sat = 0
    all_time_rl = 0
    for k,v in info_dict.items():
       if v[1] > 300:
            print(k,v[0],v[1],v[2],v[3],v[4])
            count += 1
            all_time_sat += v[1]
            all_time_rl += v[4]
            if v[5] == 'succeed':
                succeed_count += 1
                all_time_sat_succeed += v[1]
                all_time_rl_succeed += v[4]
            elif v[5] == 'failed':
                failed_count += 1
                all_time_sat_failed += v[1]
                all_time_rl_failed += v[4]
    print(count,succeed_count,failed_count,all_time_sat,all_time_rl,all_time_sat_succeed,all_time_rl_succeed,all_time_sat_failed,all_time_rl_failed)

#选择修正过的求解时间大于300s的测试数据
def test_group_1_save():
    solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_bingxing.txt'
    solve_dict = load_dictionary(solve_name)

    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1109_pre_SMTimer_save_docker_llama_3.1:70b_1200s.txt'
    info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1111_pre_SMTimer_save_docker_llama_3.1:70b_1200s_info_dict_rl.txt'
    info_dict = load_dictionary(info_name)
    for k,v in info_dict.items():
        if k in solve_dict.keys():
            info_dict[k][0] = solve_dict[k][0]
            info_dict[k][1] = solve_dict[k][1]
            info_dict[k][2] = solve_dict[k][2]
    # if not os.path.exists(info_name):
    #     # 文件不存在时，创建文件
    #     info_dict = {}
    #     with open(info_name, 'w') as file:
    #         json.dump(info_dict, file, indent=4)
    #     print(f'文件{info_name} 已创建。')
    # else:
    #     info_dict = load_dictionary(info_name)
    #     print(f'文件已存在。')
    count = 0
    succeed_count = 0
    failed_count = 0
    all_time_sat_succeed = 0
    all_time_rl_succeed = 0
    all_time_sat_failed = 0
    all_time_rl_failed = 0
    all_time_sat = 0
    all_time_rl = 0
    unknown_count = 0
    sat_count = 0
    unknown2succeed = 0
    unknown2failed = 0
    sat2succeed = 0
    sat2failed = 0
    for k,v in info_dict.items():

        if v[1] > 300:
            if v[0] == 'sat' and v[1] <= 1200:
                sat_count += 1
                if v[5] == 'succeed':
                    sat2succeed += 1
                else:
                    sat2failed += 1
            elif v[1]>1200:
                print(k,v)
                unknown_count += 1
                if v[5] == 'succeed':
                    unknown2succeed += 1
                else:
                    unknown2failed += 1
            # print(k,v[0],v[1],v[2],v[3],v[4])
            count += 1
            all_time_sat += v[1]
            all_time_rl += v[4]
            if v[5] == 'succeed':
                succeed_count += 1
                all_time_sat_succeed += v[1]
                all_time_rl_succeed += v[4]
            elif v[5] == 'failed':
                failed_count += 1
                all_time_sat_failed += v[1]
                all_time_rl_failed += v[4]
    print(f"Total Count: {count}, Succeed Count: {succeed_count}, Failed Count: {failed_count}, "
          f"All Time SAT: {all_time_sat}, All Time RL: {all_time_rl}, "
          f"All Time SAT Succeed: {all_time_sat_succeed}, All Time RL Succeed: {all_time_rl_succeed}, "
          f"All Time SAT Failed: {all_time_sat_failed}, All Time RL Failed: {all_time_rl_failed}")
    print(f"Unknown Count: {unknown_count}, SAT Count: {sat_count}, "
          f"Unknown to Succeed: {unknown2succeed}, Unknown to Failed: {unknown2failed}, "
          f"SAT to Succeed: {sat2succeed}, SAT to Failed: {sat2failed}")
    #修改求解时间,求解时间大于1200s的测试数据记为unknown
#选择修正过的求解时间大于300s的测试数据
def test_group_1_no_save():
    solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_bingxing.txt'
    solve_dict = load_dictionary(solve_name)

    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1109_pre_SMTimer_llama3.1:70b_1200s.txt'
    info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1110_pre_SMTimer_llama3.1:70b_1200s_info_dict_rl.txt'

    info_dict = load_dictionary(info_name)
    for k,v in info_dict.items():
        if k in solve_dict.keys():
            info_dict[k][0] = solve_dict[k][0]
            info_dict[k][1] = solve_dict[k][1]
            info_dict[k][2] = solve_dict[k][2]
    # if not os.path.exists(info_name):
    #     # 文件不存在时，创建文件
    #     info_dict = {}
    #     with open(info_name, 'w') as file:
    #         json.dump(info_dict, file, indent=4)
    #     print(f'文件{info_name} 已创建。')
    # else:
    #     info_dict = load_dictionary(info_name)
    #     print(f'文件已存在。')
    count = 0
    succeed_count = 0
    failed_count = 0
    all_time_sat_succeed = 0
    all_time_rl_succeed = 0
    all_time_sat_failed = 0
    all_time_rl_failed = 0
    all_time_sat = 0
    all_time_rl = 0
    unknown_count = 0
    sat_count = 0
    unknown2succeed = 0
    unknown2failed = 0
    sat2succeed = 0
    sat2failed = 0
    for k,v in info_dict.items():

        if v[1] > 300:
            if v[0] == 'sat' and v[1] <= 1200:
                sat_count += 1
                if v[4] == 'succeed':
                    sat2succeed += 1
                else:
                    sat2failed += 1
            elif v[1]>1200:
                print(k,v)
                unknown_count += 1
                if v[4] == 'succeed':
                    unknown2succeed += 1
                else:
                    unknown2failed += 1
            # print(k,v[0],v[1],v[2],v[3],v[4])
            count += 1
            all_time_sat += v[1]
            all_time_rl += v[3]
            if v[4] == 'succeed':
                succeed_count += 1
                all_time_sat_succeed += v[1]
                all_time_rl_succeed += v[3]
            elif v[4] == 'failed':
                failed_count += 1
                all_time_sat_failed += v[1]
                all_time_rl_failed += v[3]
    print(f"Total Count: {count}, Succeed Count: {succeed_count}, Failed Count: {failed_count}, "
          f"All Time SAT: {all_time_sat}, All Time RL: {all_time_rl}, "
          f"All Time SAT Succeed: {all_time_sat_succeed}, All Time RL Succeed: {all_time_rl_succeed}, "
          f"All Time SAT Failed: {all_time_sat_failed}, All Time RL Failed: {all_time_rl_failed}")
    print(f"Unknown Count: {unknown_count}, SAT Count: {sat_count}, "
          f"Unknown to Succeed: {unknown2succeed}, Unknown to Failed: {unknown2failed}, "
          f"SAT to Succeed: {sat2succeed}, SAT to Failed: {sat2failed}")
    #修改求解时间,求解时间大于1200s的测试数据记为unknown
#选择修正过的求解时间大于300s的测试数据
def test_group_1_llm():
    solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict.txt'
    solve_dict = load_dictionary(solve_name)

    info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_normal_1109_llm_no_rl_direct_solve_docker_llama3.1:70b_1set_1200s.txt'
    info_dict = load_dictionary(info_name)
    for k,v in info_dict.items():
        if k in solve_dict.keys():
            info_dict[k][0] = solve_dict[k][0]
            info_dict[k][1] = solve_dict[k][1]
            # info_dict[k][2] = solve_dict[k][2]
    # if not os.path.exists(info_name):
    #     # 文件不存在时，创建文件
    #     info_dict = {}
    #     with open(info_name, 'w') as file:
    #         json.dump(info_dict, file, indent=4)
    #     print(f'文件{info_name} 已创建。')
    # else:
    #     info_dict = load_dictionary(info_name)
    #     print(f'文件已存在。')
    count = 0
    succeed_count = 0
    failed_count = 0
    all_time_sat_succeed = 0
    all_time_rl_succeed = 0
    all_time_sat_failed = 0
    all_time_rl_failed = 0
    all_time_sat = 0
    all_time_rl = 0
    unknown_count = 0
    sat_count = 0
    unknown2succeed = 0
    unknown2failed = 0
    sat2succeed = 0
    sat2failed = 0
    for k,v in info_dict.items():

        if v[1] > 300:
            if v[0] == 'sat' and v[1] <= 1200:
                sat_count += 1
                if v[3] == 'succeed':
                    sat2succeed += 1
                else:
                    sat2failed += 1
            elif v[1]>1200:
                print(k,v)
                unknown_count += 1
                if v[3] == 'succeed':
                    unknown2succeed += 1
                else:
                    unknown2failed += 1
            # print(k,v[0],v[1],v[2],v[3],v[4])
            count += 1
            all_time_sat += v[1]

            if v[3] == 'succeed':
                succeed_count += 1
                all_time_sat_succeed += v[1]
                all_time_rl_succeed += v[6]
                all_time_rl += v[6]
            elif v[3] == 'failed':
                failed_count += 1
                all_time_sat_failed += v[1]
                all_time_rl_failed += v[4]
                all_time_rl += v[4]
    print(f"Total Count: {count}, Succeed Count: {succeed_count}, Failed Count: {failed_count}, "
          f"All Time SAT: {all_time_sat}, All Time RL: {all_time_rl}, "
          f"All Time SAT Succeed: {all_time_sat_succeed}, All Time RL Succeed: {all_time_rl_succeed}, "
          f"All Time SAT Failed: {all_time_sat_failed}, All Time RL Failed: {all_time_rl_failed}")
    print(f"Unknown Count: {unknown_count}, SAT Count: {sat_count}, "
          f"Unknown to Succeed: {unknown2succeed}, Unknown to Failed: {unknown2failed}, "
          f"SAT to Succeed: {sat2succeed}, SAT to Failed: {sat2failed}")

    print(len(info_dict),len(solve_dict))
    #修改求解时间,求解时间大于1200s的测试数据记为unknown
def test_group_2():
    info_name = 'info_dict_gai_6_normal_1109_pre_SMTimer_llama3.1:70b_1200s_result.txt'
    if not os.path.exists(info_name):
        # 文件不存在时，创建文件
        info_dict = {}
        with open(info_name, 'w') as file:
            json.dump(info_dict, file, indent=4)
        print(f'文件{info_name} 已创建。')
    else:
        info_dict = load_dictionary(info_name)
        print(f'文件已存在。')
    count = 0
    succeed_count = 0
    failed_count = 0
    all_time_sat = 0
    all_time_rl = 0
    for k,v in info_dict.items():
       if v[1] > 1200:
            # info_dict[k][0] = 'unknown'
            print(k,v[0],v[1],v[2],v[3],v[4])
            count += 1
    print(count)
    # with open(info_name, 'w') as file:
    #     json.dump(info_dict, file, indent=4)
    # 修改求解时间,求解时间大于1200s的测试数据记为unknown
def test_group_3():
    with open('/home/lz/sibyl_3/src/networks/info_dict_rl.txt', 'r') as file:
        rl_dict = json.load(file)
    info_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict.txt'
    if not os.path.exists(info_name):
        # 文件不存在时，创建文件
        info_dict = {}
        with open(info_name, 'w') as file:
            json.dump(info_dict, file, indent=4)
        print(f'文件{info_name} 已创建。')
    else:
        info_dict = load_dictionary(info_name)
        print(f'文件已存在。')
    count = 0
    succeed_count = 0
    failed_count = 0
    all_time_sat = 0
    all_time_rl = 0
    for k, v in info_dict.items():
        if (v[0]=='sat' or v[0]=='unknown') and v[1] > 300 and k in rl_dict.keys():
            # info_dict[k][0] = 'unknown'
            # print(k, v[0], v[1], v[2], v[3], v[4])
            print(k,v)
            count += 1
    print(count)
    # with open(info_name, 'w') as file:
    #     json.dump(info_dict, file, indent=4)
def extract_unique_keys(file_paths):
    unique_keys = set()  # 使用集合来存储不重复的键

    for file_path in file_paths:
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)  # 加载JSON数据
                unique_keys.update(data.keys())  # 更新集合，添加新的键
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {file_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    return unique_keys
def something():
    solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict.txt'
    solve_dict = load_dictionary(solve_name)

    info_name = '/home/lz/sibyl_3/src/networks/info_dict_rl.txt'
    info_dict = load_dictionary(info_name)

    count = 0
    for k,v in info_dict.items():
        if (info_dict[k][0] == 'sat' or info_dict[k][0] == 'unknown') and info_dict[k][1] > 300:
            count += 1
        # if k in solve_dict.keys():
        #     if (solve_dict[k][0] == 'sat' or solve_dict[k][0] == 'unknown') and solve_dict[k][1] > 300:
        #         count += 1
    print(len(solve_dict),len(info_dict),count)
    # 定义文件路径列表
    file_paths = [
        '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_normal_1103_llm_no_rl_direct_solve_docker_llama3.1:70b_1set.txt',
        '/home/lz/sibyl_3/src/networks/info_dict_rl.txt',
        '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_109_2_SMTimer.txt',
        # '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict.txt'
    ]

    # 调用函数并打印结果
    unique_keys = extract_unique_keys(file_paths)
    print("Unique keys across all files:")
    print(len(unique_keys))
    count = 0
    for k in unique_keys:
        if k in solve_dict.keys():
            count += 1
    print(count)
    # for key in sorted(unique_keys):  # 排序输出
    #     print(key)

    solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_normal_1109_llm_no_rl_direct_solve_docker_llama3.1:70b_1set_1200s.txt'
    solve_dict = load_dictionary(solve_name)

    info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1110_pre_SMTimer_llama3.1:70b_1200s_info_dict_rl.txt'
    info_dict = load_dictionary(info_name)
    # 提取字典的键
    solve_keys = set(solve_dict.keys())
    info_keys = set(info_dict.keys())

    # 计算交集和并集
    keys_intersection = solve_keys.intersection(info_keys)
    keys_union = solve_keys.union(info_keys)

    print(len(solve_dict),len(info_dict))
    # 打印结果
    print("Keys Intersection:", len(keys_intersection))

    print("Keys Union:", len(keys_union))
    print(keys_union)

    info_dict = load_dictionary('/home/lz/PycharmProjects/Pearl/test_rl/result_dict.txt')
    print(len(info_dict))
def test_group_2_no_save(solve_name,info_name):

    solve_dict = load_dictionary(solve_name)

    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1109_pre_SMTimer_llama3.1:70b_1200s.txt'

    info_dict = load_dictionary(info_name)
    for k,v in info_dict.items():
        if k in solve_dict.keys():
            info_dict[k][0] = solve_dict[k][0]
            info_dict[k][1] = solve_dict[k][1]
            info_dict[k][2] = solve_dict[k][2]
    # if not os.path.exists(info_name):
    #     # 文件不存在时，创建文件
    #     info_dict = {}
    #     with open(info_name, 'w') as file:
    #         json.dump(info_dict, file, indent=4)
    #     print(f'文件{info_name} 已创建。')
    # else:
    #     info_dict = load_dictionary(info_name)
    #     print(f'文件已存在。')
    result_dict = {}
    result_dict['succeed'] = []
    result_dict['failed'] = []
    result_dict['unknown to succeed'] = []
    result_dict['unknown to failed'] = []
    result_dict['sat to succeed'] = []
    result_dict['sat to failed'] = []
    count = 0
    succeed_count = 0
    failed_count = 0
    all_time_sat_succeed = 0
    all_time_rl_succeed = 0
    all_time_sat_failed = 0
    all_time_rl_failed = 0
    all_time_sat = 0
    all_time_rl = 0
    unknown_count = 0
    sat_count = 0
    unknown2succeed = 0
    unknown2failed = 0
    sat2succeed = 0
    sat2failed = 0
    for k,v in info_dict.items():

        if v[1] > 300:
            if v[0] == 'sat' and v[1] <= 1200:
                sat_count += 1
                if v[4] == 'succeed':
                    result_dict['succeed'].append(k)
                    sat2succeed += 1
                    result_dict['sat to succeed'].append(k)
                else:
                    result_dict['failed'].append(k)
                    sat2failed += 1
                    result_dict['sat to failed'].append(k)
            elif v[1]>1200:
                # print(k,v)
                unknown_count += 1
                if v[4] == 'succeed':
                    result_dict['succeed'].append(k)
                    unknown2succeed += 1
                    result_dict['unknown to succeed'].append(k)
                else:
                    result_dict['failed'].append(k)
                    unknown2failed += 1
                    result_dict['unknown to failed'].append(k)
            # print(k,v[0],v[1],v[2],v[3],v[4])
            count += 1
            all_time_sat += v[1]
            all_time_rl += v[3]
            if v[4] == 'succeed':
                succeed_count += 1
                all_time_sat_succeed += v[1]
                all_time_rl_succeed += v[3]
            elif v[4] == 'failed':
                failed_count += 1
                all_time_sat_failed += v[1]
                all_time_rl_failed += v[3]
    # logger.info(result_dict)
    print(f"Total Count: {count}, Succeed Count: {succeed_count}, Failed Count: {failed_count}, "
          f"All Time SAT: {all_time_sat}, All Time RL: {all_time_rl}, "
          f"All Time SAT Succeed: {all_time_sat_succeed}, All Time RL Succeed: {all_time_rl_succeed}, "
          f"All Time SAT Failed: {all_time_sat_failed}, All Time RL Failed: {all_time_rl_failed}")
    print(f"Unknown Count: {unknown_count}, SAT Count: {sat_count}, "
          f"Unknown to Succeed: {unknown2succeed}, Unknown to Failed: {unknown2failed}, "
          f"SAT to Succeed: {sat2succeed}, SAT to Failed: {sat2failed}")
    print(result_dict)
    #修改求解时间,求解时间大于1200s的测试数据记为unknown
if __name__ == '__main__':
    # test_group()
    # test_group_1()
    # test_group_2()
    # test_group_3()
    # test_group_1_save()
    # test_group_1_no_save()
    # test_group_1_llm()
    # something()

    # solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_bingxing.txt'
    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1110_pre_SMTimer_llama3.1:70b_1200s_info_dict_rl.txt'
    # test_group_2_no_save(solve_name,info_name)


    solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_smt_comp.txt'
    info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1118_pre_llm_llama3.1:70b_1200s_QF_LIA.txt'
    test_group_2_no_save(solve_name,info_name)
