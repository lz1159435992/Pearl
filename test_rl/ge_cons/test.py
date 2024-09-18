from z3 import *
import time
import json
from test_rl.test_script.utils import normalize_smt_str


def simple_hash(x):
    return x * 37 % 100


def solve_and_measure_time(solver, timeout):
    solver.set("timeout", timeout)
    start_time = time.time()
    result = solver.check()
    elapsed_time = time.time() - start_time
    if result == sat:
        return "求解成功", solver.model(), elapsed_time
    elif result == unknown:
        return "求解超时", None, elapsed_time
    else:
        return "求解失败", None, elapsed_time


def main():
    # 创建求解器
    with open('smt2.txt', 'r') as file:
        smtlib_str = file.read()
    assertions = parse_smt2_string(smtlib_str)
    solver = Solver()  # 定义变量
    for assertion in assertions:
        solver.add(assertion)

    smtlib_str, sorted_variable_dict, a_list = normalize_smt_str(smtlib_str)
    print(smtlib_str)
    print(sorted_variable_dict)
    print(a_list)
    # 添加更多复杂的约束
    # solver.add(Or(x + y + z == 10, x - y - z == 10))
    # solver.add(And(x * y * z < 1000, x * y * z > 100))
    # solver.add(If(x % 2 == 0, x + y < 50, x + y > 50))
    # solver.add(If(y % 3 == 0, y - x > 10, y - x < -10))
    # solver.add(x**3 + y**3 == z**3)
    timeout = 10

    # timeout = 12000000  # 尝试求解，并测量时间
    result, model, time_taken = solve_and_measure_time(solver, timeout)
    print(f"未具体化变量 - 结果: {result}, 时间: {time_taken:.2f} 秒, 模型: {model}")  # 清除之前的约束

    time_taken = 1743.46

    output_file = 'auto_gen_v2.txt'
    if not os.path.exists(output_file):
        # 文件不存在时，创建文件
        info_dict = {}
        with open(output_file, 'w') as file:
            json.dump(info_dict, file, indent=4)
        print(f"文件 {output_file} 已创建。")
    if time_taken > 1000:
        with open(output_file, 'r') as file:
            result_dict = json.load(file)
        if solver.to_smt2() not in result_dict.values():
            result_dict[len(result_dict)] = [time_taken, solver.to_smt2()]
        else:
            print(f"求解结果已存在。")
        with open(output_file, 'w') as file:
            json.dump(result_dict, file, indent=4)


if __name__ == "__main__":
    main()
