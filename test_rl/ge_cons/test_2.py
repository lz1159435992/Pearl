
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
    # solver.add(((x * 12) % 100)**2 + y**2 > 50)
    # solver.add(If(x > y, z**2 == 16, z**2 < 10))
    # solver.add(If(((z * 37) % 100) > 30, y**2 == x + 10, y**2 == x - 5))
    # solver.add(x * y - z != 0) # 2分钟超时限制（单位：毫秒）

#075
    x = BitVec('x', 8)
    y = BitVec('y', 8)
    z = BitVec('z', 8)  # 添加复杂的约束
    # # x = Int('x')
    # # y = Int('y')
    # # z = Int('z') # 添加复杂的约束
    solver.add(((x * 37) % ((x * 12) % 100)) + y == ((z * ((x * 12) % 7)) % 8))
    # # solver.add(x * x + y * y > (z * 396) % ((x * 6) % 71) * (z * 396) % ((x * 6) % 71) + 100)
    # # solver.add(((x * 12) % 100) * ((x * 12) % 100) + y * y > 50)
    # # solver.add(If(x > y, z * z == 16, z * z < 10))
    # # solver.add(If(((z * 37) % 100) > 30, y * y == x + 10, y * y == x - 5))
    # # solver.add(x * y - z != 0)  # 2分钟超时限制（单位：毫秒）
    #
    # # Add more complex modulus and multiplication constraints




    # 使用自定义的位向量幂运算代替 **

    complex_expr = ((((x * 6) ^ (x >> 2)) & 0xFF) * ((x | 0x3) % 73) + ((x & 0x1F) << 3)) % 71/2


    base = complex_expr
    exp = complex_expr
    solver.add(((x*x*y*z) % ((x*y * 12) % (x % y + z))) * ((x * z) % ((x * 128) % 7)) + y*y == complex_expr * z * ((x * z) % ((x * x) % (x - 7 * y + z))) * ((z * ((x * 12) % 7)) % 8) * ((x * 3) % ((x * 128) % 7)))
    solver.add(x * x + y * y > (z * 396) % ((x * 6) % 71) * (z * 396) % ((x * 6) % 71) + 100)
    #
    # # Introduce more bitwise operations to increase complexity
    # solver.add(((x * 12) % 17) * ((x * 12) % 41) + (y ^ z) * (y & z) > (z * 396) % ((x * 6) % 71))
    #
    # # Add nested conditionals to increase difficulty
    # solver.add(z3.If(x > y, z * z == 16, z * z < 10))
    # solver.add(z3.If(((z * 37) % 100) > 30, y * y == (x + 10) & z, y * y == x - 5))
    #
    # # Increase the range of interdependence between variables
    # solver.add((x * x * y - z) ^ (x * z + y) != 0)
    #
    # # Add a non-linear combination involving multiple variables
    # solver.add(z * (x | y) - (x & z) + (y >> 1) > 12345)


    # 更复杂的表达式，但实际赋值后易于简化
    hash1 = (((x ^ (y << 3)) + z) % 256 + ((x & 0xF) << 4) + ((z & 0xF) << 2)) % 512

    # 引入更多的位运算和掩码，增加看似复杂的结构
    hash2 = ((((z ^ (x >> 2)) * y) & 0xFF) ^ ((x << 2) | (z >> 3))) & 0xFFFF

    # 进一步组合两个“复杂”的哈希函数，但其中实际计算不复杂
    solver.add((hash1 ^ hash2) == ((x * y - z) & 0x3FF))

    solver.add(((x << 5) ^ (y >> 3) ^ (z * 31)) & 0xFFFFFFFF > (x * y + z) % 100)
    solver.add(((x * 13 + y * 17) % 256) ^ ((z * 29) % 512) > (x * y) % 1024)


    # x = z3.BitVec('x', 16)
    # y = z3.BitVec('y', 16)
    # z = z3.BitVec('z', 16)
    #
    # solver = z3.Solver()

    # # 引入类似哈希的位操作和模运算
    # solver.add(((x << 5) ^ (y >> 3) ^ (z * 31)) & 0xFFFFFFFF == (x * y + z) % 100)
    # solver.add(((x * 13 + y * 17) % 256) ^ ((z * 29) % 512) == (x * y) % 1024)
    #
    # # 使用循环位移和更多的非线性组合
    # solver.add(((z << (x % 16)) | (z >> (16 - (x % 16)))) == ((x * 23) ^ (y * z)) % 100)

    # # 组合多个哈希风格操作
    # hash1 = ((x ^ (y << 3)) + z) % 256
    # hash2 = ((z ^ (x >> 2)) * y) % 512
    # solver.add(hash1 ^ hash2 == (x * y - z) % 1024)

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