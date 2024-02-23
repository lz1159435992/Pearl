import json
import os
import time
from z3 import *
value_dict = {}
result_dict = {}
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
    result_dict[filepath] = result_list
    if model:
        for d in model.decls():
            if str(model[d]) not in value_dict.keys():
                value_dict[str(model[d])] = 1
            else:
                value_dict[str(model[d])] += 1
    print(f"result: {result}, time: {time_taken:.2f} value: {model},filepath:{filepath}")

if __name__ == '__main__':
    solve('/home/yy/Downloads/smt/buzybox_angr.tar.gz/single_test/gnuzip/gunzip1159114',30000)




