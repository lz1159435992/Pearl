
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

    # 复杂的模运算、位运算和条件分支，难以求解
    solver.add((x ^ (y * z) + (z * 57 % (x + 2))) * (y & (x | z)) % 1024 == (x * y) % 511)

    # 额外的条件分支，当 x, y, z 取某些值时，这些分支会变得简单
    solver.add(If((x + y) % 256 == 128, z * (x + y) % 97 == 15, z * (x - y) % 97 == 42))

    # 增加一些位运算和非线性组合
    solver.add(((x * y * z) % (x + 1)) * ((x ^ y) + (z >> 1)) > 5000)

    # 添加依赖于某些具体值的条件，使问题在赋值后简化
    solver.add(If(x == 123456 and y == 789012 and z == 345678, x + y + z == 1253456, x + y + z > 1000000))

    # 添加一个复杂的等式约束，使问题在未赋值时难解
    solver.add(((x * y * z) % (x * y + 1)) * ((z ^ (x + y)) % 97) == 0)


    # complex_expr = ((((x * 6) ^ (x >> 2)) & 0xFF) * ((x | 0x3) % 73) + ((x & 0x1F) << 3)) % 71 / 2
    #
    # base = complex_expr
    # exp = complex_expr
    solver.add(
        ((x * x * y * z) % ((x * y * 12) % (x % y + z))) * ((x * z) % ((x * 128) % 7)) + y * y == (((((x * 6) ^ (x >> 2)) & 0xFF) * ((x | 0x3) % 73) + ((x & 0x1F) << 3)) % 71 / 2) * (((((x * 6) ^ (x >> 2)) & 0xFF) * ((x | 0x3) % 73) + ((x & 0x1F) << 3)) % 71 / 2) * z * (
                    (x * z) % ((x * x) % (x - 7 * y + z))) * ((z * ((x * 12) % 7)) % 8) * ((x * 3) % ((x * 128) % 7)))


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