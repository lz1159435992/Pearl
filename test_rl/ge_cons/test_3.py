
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

    # x = Int('x')
    # y = Int('y')
    # z = Int('z') # 添加复杂的约束
    # solver.add(((x * 37) % 100) + y == ((z * 396) % 100))
    # solver.add(x**2 + y**2 > 50)
    # solver.add(((x * 12) % 100) > 50)
    # solver.add(If(x > y, z**2 == 16, z**2 < 10))
    # solver.add(If(((z * 37) % 100) > 30, y**2 == x + 10, y**2 == x - 5))
    # solver.add(x * y - z != 0) # 2分钟超时限制（单位：毫秒）
    #
    # timeout = 12000000 # 尝试求解，并测量时间
    # result, model, time_taken = solve_and_measure_time(solver, timeout)
    # print(f"未具体化变量 - 结果: {result}, 时间: {time_taken:.2f} 秒, 模型: {model}") # 清除之前的约束
    # solver.reset() # 再次定义变量和约束


    x = Int('x')
    y = Int('y')
    z = Int('z')  # 添加复杂的约束
    solver.add(((x * 37) % 100) + y == ((z * 396) % 100))
    solver.add(x ** 2 + y ** 2 > 50)
    solver.add(((x * 12) % 100) > 50)
    solver.add(If(x > y, z ** 2 == 16, z ** 2 < 10))
    solver.add(If(((z * 37) % 100) > 30, y ** 2 == x + 10, y ** 2 == x - 5))
    solver.add(x * y - z != 0)  # 2分钟超时限制（单位：毫秒）

    start_time = time.time()
    result2 = solver.check()
    elapsed_time = time.time() - start_time
    print(result2, elapsed_time)
    start_time = time.time()
    result2 = solver.check()
    elapsed_time = time.time() - start_time
    print(result2, elapsed_time) # 具体化一个变量
    solver.add(x == 2199)
    start_time = time.time()
    result2 = solver.check()
    elapsed_time = time.time() - start_time
    print(result2, elapsed_time)
    print(result2) # 再次尝试求解，并测量时间
    result, model, time_taken = solve_and_measure_time(solver, timeout)
    print(f"具体化变量 z - 结果: {result}, 时间: {time_taken:.2f} 秒, 模型: {model}")
if __name__ == "__main__":
    main()