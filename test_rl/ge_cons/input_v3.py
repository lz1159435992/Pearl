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
    if result == sat: return "求解成功", solver.model(), elapsed_time
    elif result == unknown: return "求解超时", None, elapsed_time
    else: return "求解失败", None, elapsed_time
def main():
# 创建求解器
    solver = Solver() # 定义变量

    x = BitVec('x', 32)
    y = BitVec('y', 32)
    z = BitVec('z', 32)

    # 定义常量为位向量
    const_91 = BitVecVal(91, 32)
    const_13 = BitVecVal(13, 32)
    const_4096 = BitVecVal(4096, 32)
    const_255 = BitVecVal(255, 32)
    const_512 = BitVecVal(512, 32)
    const_75 = BitVecVal(75, 32)
    const_10 = BitVecVal(10, 32)
    const_4_bit = BitVecVal(4, 32)
    const_2A = BitVecVal(0x2A, 32)
    const_9 = BitVecVal(9, 32)
    const_31 = BitVecVal(31, 32)
    const_500 = BitVecVal(500, 32)
    const_123 = BitVecVal(123, 32)
    const_3 = BitVecVal(3, 32)

    # 增加复杂的位运算和模运算
    solver.add((x * y ^ z + (x | const_91) % (y & const_13)) * (z ^ (x >> const_4_bit)) % const_4096 == (x * z) % const_255)

    # 增加条件分支，当 x, y, z 取特定值时，约束会简化
    solver.add(If((x * y + z) % const_512 == const_75, (x * z) % const_10 == const_4_bit, (y ^ z) % const_13 == const_2A))

    # 增加一些复杂的位移和组合操作
    solver.add(((x ^ y) * (z >> 2) + (x & y) % const_9) * ((z | const_31) % const_500) > const_123)

    # 添加复杂条件分支和非线性表达式，使得在赋值前难以求解
    solver.add(If(x == const_91 and y == const_123 and z == const_255,
                  x + y - z == const_3,
                  (x ^ y ^ z) % const_512 > const_9))

    # 增加带有非线性运算和复杂位移的等式约束
    solver.add(((x * y + z) % (x ^ y + z * 3)) * ((x & y) + (z >> const_3)) == 0)

    # 复杂表达式
    complex_expr_2 = ((((z * const_91) + (y >> const_2A)) ^ const_13) * ((x & const_255) % const_500) + (
                (y | const_31) << const_4_bit)) % const_4096 / 2

    solver.add(
        ((x ^ y * z) % ((x * y * const_75) % (z + const_31))) * ((y * z) % ((z * const_512) % const_9)) + y ==
        (complex_expr_2 * complex_expr_2) % ((z * ((x + const_4_bit) % const_31)) % const_500))




#具体化
    # solver.add(z == -2) # 2分钟超时限制（单位：毫秒）
    print(solver.to_smt2())
    # timeout = 10

    timeout = 12000000 # 尝试求解，并测量时间
    result, model, time_taken = solve_and_measure_time(solver, timeout)
    print(f"未具体化变量 - 结果: {result}, 时间: {time_taken:.2f} 秒, 模型: {model}") # 清除之前的约束


    # time_taken = 10000
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