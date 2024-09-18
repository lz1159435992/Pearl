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


    # 新的复杂模运算和条件分支
    solver.add(((x + (y ^ z)) * (x - y) + (z % (y + 3))) * (x | (z & y)) % 2048 == (x * y * z) % 1023)

    # 当特定条件满足时，问题变得简单
    solver.add(If((x * z) % 512 == 256, y * (x - z) % 89 == 22, (x ^ y ^ z) % 89 == 77))

    # 增加复杂的位运算和非线性组合
    solver.add(((x | y | z) % (x + y + 2)) * ((x & y) ^ (z >> 2)) < 10000)

    # 增加依赖特定值的条件，简化后的求解更容易
    solver.add(If(x == 654321 and y == 123456 and z == 987654, x + y - z == -210877, x * y * z > 500000))

    # 更复杂的位运算和模运算
    solver.add(((x ^ y * z) % (y + z + 1)) * ((x | z) & (y >> 1)) % 4096 == ((x * z) + (y ^ z)) % 1024)

    # 更复杂的条件和表达式组合
    solver.add(
        ((x * x * z) % ((y * 5) % (z + x))) * ((x & y) ^ (z >> 3)) + z * z == (
                ((((x * 7) ^ (z >> 4)) & 0xFF) * ((z | 0x9) % 53) + ((z & 0x3F) << 2)) % 67 / 3) * (
                ((((y * 6) ^ (x >> 1)) & 0xEF) * ((y | 0x7) % 97) + ((x & 0x1A) << 4)) % 73 / 2) * z * (
                (y * z) % ((y * x) % (x - 5 * z + y))) * ((x * ((z * 5) % 9)) % 10) * ((y * 2) % ((y * 99) % 11)))


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
        with open('auto_gen.txt', 'w') as file:
            json.dump(result_dict, file, indent=4)
if __name__ == "__main__":
    main()