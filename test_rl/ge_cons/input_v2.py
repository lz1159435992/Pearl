import json

from z3 import *
import time


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
    solver = Solver()  # 定义变量

    x = BitVec('x', 32)
    y = BitVec('y', 32)
    z = BitVec('z', 32)

    # 定义常量为位向量
    const_57 = BitVecVal(57, 32)
    const_2 = BitVecVal(2, 32)
    const_1024 = BitVecVal(1024, 32)
    const_511 = BitVecVal(511, 32)
    const_256 = BitVecVal(256, 32)
    const_128 = BitVecVal(128, 32)
    const_97 = BitVecVal(97, 32)
    const_15 = BitVecVal(15, 32)
    const_42 = BitVecVal(42, 32)
    const_123456 = BitVecVal(123456, 32)
    const_789012 = BitVecVal(789012, 32)
    const_345678 = BitVecVal(345678, 32)
    const_1253456 = BitVecVal(1253456, 32)
    const_1000000 = BitVecVal(1000000, 32)
    const_5000 = BitVecVal(5000, 32)
    const_6 = BitVecVal(6, 32)
    const_2_bit = BitVecVal(2, 32)
    const_FF = BitVecVal(0xFF, 32)
    const_3 = BitVecVal(3, 32)
    const_73 = BitVecVal(73, 32)
    const_1F = BitVecVal(0x1F, 32)
    const_3_bit = BitVecVal(3, 32)
    const_71 = BitVecVal(71, 32)
    const_12 = BitVecVal(12, 32)
    const_128 = BitVecVal(128, 32)
    const_7 = BitVecVal(7, 32)
    const_8 = BitVecVal(8, 32)

    solver.add(
        (x ^ (y * z) + (z * const_57 - (x + const_2) * (z * const_57 // (x + const_2)))) * (
                    y & (x | z)) - const_1024 * (
                (x ^ (y * z) + (z * const_57 - (x + const_2) * (z * const_57 // (x + const_2)))) * (
                y & (x | z)) // const_1024) == (x * y - const_511 * (x * y // const_511)))

    solver.add(If((x + y) - const_256 * ((x + y) // const_256) == const_128,
                  z * (x + y) - const_97 * (z * (x + y) // const_97) == const_15,
                  z * (x - y) - const_97 * (z * (x - y) // const_97) == const_42))

    solver.add(((x * y * z) - (x + 1) * ((x * y * z) // (x + 1))) * ((x ^ y) + (z >> 1)) > const_5000)

    solver.add(((x * y * z) - (x * y + 1) * ((x * y * z) // (x * y + 1))) * (
            (z ^ (x + y)) - const_97 * ((z ^ (x + y)) // const_97)) == 0)

    complex_expr = ((((x * const_6) ^ (x >> const_2_bit)) & const_FF) * (
            (x | const_3) - const_73 * ((x | const_3) // const_73)) + ((x & const_1F) << const_3_bit)) - const_71 * ((((
                                                                                                                               (
                                                                                                                                       x * const_6) ^ (
                                                                                                                                       x >> const_2_bit)) & const_FF) * (
                                                                                                                              (
                                                                                                                                      x | const_3) - const_73 * (
                                                                                                                                      (
                                                                                                                                              x | const_3) // const_73)) + (
                                                                                                                              (
                                                                                                                                      x & const_1F) << const_3_bit)) // const_71) / 2

    solver.add(
        ((x * x * y * z) - ((x * y * const_12) - ((x % y + z) * ((x * y * const_12) // (x % y + z))))) * (
                (x * z) - ((x * const_128) - (const_7 * (x * const_128) // const_7))) + y * y ==
        (complex_expr * complex_expr) * z * (
                (x * z) - ((x * x) - ((x - const_7 * y + z) * (x * x // (x - const_7 * y + z))))) *
        ((z * ((x * const_12) - (const_7 * (x * const_12) // const_7))) - (
                    const_8 * (z * ((x * const_12) // const_7))) * (
                 (x * const_3) - ((x * const_128) - (const_7 * (x * const_128) // const_7))))
    )

    #具体化
    # solver.add(z == -2) # 2分钟超时限制（单位：毫秒）
    print(solver.to_smt2())
    timeout = 12000000  # 尝试求解，并测量时间
    result, model, time_taken = solve_and_measure_time(solver, timeout)
    print(f"未具体化变量 - 结果: {result}, 时间: {time_taken:.2f} 秒, 模型: {model}")  # 清除之前的约束

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
