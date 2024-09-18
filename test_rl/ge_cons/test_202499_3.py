
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


    # 定义位向量
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
    const_FF = BitVecVal(0xFF, 32)
    const_3 = BitVecVal(3, 32)
    const_73 = BitVecVal(73, 32)
    const_1F = BitVecVal(0x1F, 32)
    const_71 = BitVecVal(71, 32)
    const_12 = BitVecVal(12, 32)
    const_128 = BitVecVal(128, 32)
    const_7 = BitVecVal(7, 32)
    const_8 = BitVecVal(8, 32)
    # 复杂的模运算、位运算和条件分支，难以求解
    solver.add((x ^ (y * z) + (z * const_57 % (x + const_2))) * (y & (x | z)) % const_1024 == (x * y) % const_511)

    # 额外的条件分支，当 x, y, z 取某些值时，这些分支会变得简单
    solver.add(If((x + y) % const_256 == const_128, z * (x + y) % const_97 == const_15, z * (x - y) % const_97 == const_42))

    # 增加一些位运算和非线性组合
    solver.add(((x * y * z) % (x + 1)) * ((x ^ y) + (z >> 1)) > const_5000)

    # 添加依赖于某些具体值的条件，使问题在赋值后简化
    solver.add(If(x == const_123456 and y == const_789012 and z == const_345678,
                  x + y + z == const_1253456,
                  x + y + z > const_1000000))

    # 添加一个复杂的等式约束，使问题在未赋值时难解
    solver.add(((x * y * z) % (x * y + const_2)) * ((z ^ (x + y)) % const_97) == 0)

    # 复杂表达式
    complex_expr = ((((x * const_6) ^ (x >> const_2)) & const_FF) * ((x | const_3) % const_73) + (
                (x & const_1F) << const_3)) % const_71 / 2

    solver.add(
        ((x * x * y * z) % ((x * y * const_12) % (x % y + z))) * ((x * z) % ((x * const_128) % const_7)) + y * y ==
        (complex_expr * complex_expr) * z * ((x * z) % ((x * x) % (x - const_7 * y + z))) *
        ((z * ((x * const_12) % const_7)) % const_8) * ((x * const_3) % ((x * const_128) % const_7)))


#具体化
    # solver.add(z == -2) # 2分钟超时限制（单位：毫秒）
    print(solver.to_smt2())
    timeout = 12000000 # 尝试求解，并测量时间
    result, model, time_taken = solve_and_measure_time(solver, timeout)
    print(f"未具体化变量 - 结果: {result}, 时间: {time_taken:.2f} 秒, 模型: {model}") # 清除之前的约束
    solver.reset() # 再次定义变量和约束
    # x = Int('x')
    # y = Int('y')
    # z = Int('z')
    # solver.add(((x * 37) % 100) + y == z)
    # solver.add(x**2 + y**2 > 50)
    # solver.add(If(x > y, z**2 == 16, z**2 < 10))
    # solver.add(If(((z * 37) % 100) > 30, y**2 == x + 10, y**2 == x - 5))
    # start_time = time.time()
    # result2 = solver.check()
    # elapsed_time = time.time() - start_time
    # print(result2, elapsed_time)
    # start_time = time.time()
    # result2 = solver.check()
    # elapsed_time = time.time() - start_time
    # print(result2, elapsed_time) # 具体化一个变量
    # solver.add(x * y - z != 0)
    # start_time = time.time()
    # result2 = solver.check()
    # elapsed_time = time.time() - start_time
    # print(result2, elapsed_time)
    # print(result2) # 再次尝试求解，并测量时间
    # result, model, time_taken = solve_and_measure_time(solver, timeout)
    # print(f"具体化变量 z - 结果: {result}, 时间: {time_taken:.2f} 秒, 模型: {model}")
if __name__ == "__main__":
    main()