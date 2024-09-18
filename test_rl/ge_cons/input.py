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
    p = BitVec('p', 32)
    q = BitVec('q', 32)
    r = BitVec('r', 32)

    # 复杂的模运算和位运算的组合，增加求解难度
    solver.add((p * (q ^ r) + (p * 41 % (r + 3))) * (q & (p | r)) % 4096 == (p * q) % 1023)

    # 条件分支，具体化时简化
    solver.add(If((p - q + r) % 128 == 64, p * (q + r) % 101 == 33, p * (q - r) % 101 == 59))

    # 增加非线性组合，影响求解时间
    solver.add(((p * q * r) % (q + 3)) * ((p ^ r) + (q >> 2)) > 20000)

    # 添加依赖具体值的条件，使得具体化时问题变得简单
    solver.add(If(p == 567890 and q == 123456 and r == 654321, p + q + r == 1345677, p + q + r > 1500000))

    # 复杂的等式约束，未赋值时难解
    solver.add(((p * q * r) % (p * q + 2)) * ((r ^ (p + q)) % 113) == 2)

    # 结合更多位运算、模运算和条件分支，增加复杂性
    solver.add(
        ((p * p * q * r) % ((p * q * 7) % (p % q + r))) * ((p * r) % ((p * 256) % 13)) + q * q == (
                ((((p * 7) ^ (p >> 3)) & 0xFF) * ((p | 0x6) % 103) + ((p & 0x3F) << 4)) % 97 / 2) * (
                ((((p * 7) ^ (p >> 3)) & 0xFF) * ((p | 0x6) % 103) + ((p & 0x3F) << 4)) % 97 / 2) * r * (
                (p * r) % ((p * p) % (p - 5 * q + r))) * ((r * ((p * 18) % 11)) % 12) * ((p * 5) % ((p * 256) % 11)))






#具体化
    # solver.add(z == -2) # 2分钟超时限制（单位：毫秒）
    print(solver.to_smt2())
    timeout = 12000000 # 尝试求解，并测量时间
    result, model, time_taken = solve_and_measure_time(solver, timeout)
    print(f"未具体化变量 - 结果: {result}, 时间: {time_taken:.2f} 秒, 模型: {model}") # 清除之前的约束


    if not os.path.exists('auto_gen.txt'):
        # 文件不存在时，创建文件
        info_dict = {}
        with open('auto_gen.txt', 'w') as file:
            json.dump(info_dict, file, indent=4)
        print(f"文件 auto_gen.txt 已创建。")
    if time_taken > 1000:
        with open('auto_gen.txt', 'r') as file:
            result_dict = json.load(file)
        if solver.to_smt2() not in result_dict.values():
            result_dict[len(result_dict)] = solver.to_smt2()
        else:
            print(f"求解结果已存在。")
        with open('auto_gen.txt', 'w') as file:
            json.dump(result_dict, file, indent=4)
if __name__ == "__main__":
    main()