import json
from z3 import *

from test_rl.test_script.search_test import solve_and_measure_time


def test_solver(file_path):
    timeout = 3000000
    with open(file_path) as file:
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
    print(result)
    print(model)
    print(time_taken)
if __name__ == '__main__':
    test_solver('/home/nju/Downloads/smt/gnu_angr.tar.gz/single_test/seq/seq140268')