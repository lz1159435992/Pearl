import json
from z3 import *
import math
import re


def preprocess_list(value_list):
    # Replace NaN values with None
    processed_list = []
    for item in value_list:
        if isinstance(item, float) and math.isnan(item):
            processed_list.append(None)
        else:
            processed_list.append(item)
    return processed_list


# Example usage
def model_to_dict(model):
    result = {}
    for var in model:
        result[str(var)] = str(model[var])
    return result


def load_dictionary(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


# 假设我们有一个大型的 SMT-LIB v2 字符串

# 分段解析字符串
def parse_smt2_in_parts(smt2_string):
    solver = Solver()
    # 使用 StringIO 来模拟文件读取
    from io import StringIO
    assertions = None
    stream = StringIO(smt2_string)
    # 逐行读取并解析
    for line in stream:
        # 忽略空行和注释
        if line.strip() and not line.strip().startswith(';'):
            if assertions is None:
                assertions = parse_smt2_string(line)
            else:
                mid = parse_smt2_string(line)
                for m in mid:
                    assertions.add(m)
    # 返回解析后的求解器
    return assertions


# 使用分段解析函数


# 一个非常长的 SMT-LIB 格式的字符串示例
long_smt_str = """
(assert (> x 5))
(assert (< y 10))
; 更多断言...
"""


# 创建一个求解器实例

# 分段解析并添加断言到求解器的函数
def parse_and_add_assertions(long_smt_str):
    solver = Solver()
    # 使用正则表达式找到所有的断言
    pattern = r'\(assert [^\)]+\)'
    assertions = re.findall(pattern, long_smt_str)
    # 遍历找到的断言字符串
    for assertion_str in assertions:
        # 因为parse_smt2_string需要完整的SMT-LIB命令，我们需要确保每个断言都被正确格式化
        formatted_assertion_str = assertion_str
        # 解析并添加断言
        assertion = parse_smt2_string(formatted_assertion_str, decls={})
        for a in assertion:
            solver.add(a)
    return solver


# 调用函数，分段解析并添加断言


# 处理声明和断言的函数
def process_smt_lib_string(smt_str):
    # 分割字符串为单独的行
    # 创建一个求解器实例
    solver = Solver()

    lines = smt_str.split('\n')
    # 用于存储声明和断言的字符串
    declarations_str = ""
    assertions_str = ""
    # 区分声明和断言
    for line in lines:
        if line.startswith("(declare"):
            declarations_str += line + "\n"
        elif line.startswith("(assert"):
            assertions_str += line + "\n"

    # 处理声明
    if declarations_str:
        # 直接解析声明
        parse_smt2_string(declarations_str, decls={})
    # 处理断言
    if assertions_str:
        # 使用正则表达式找到所有的断言
        pattern = r'\(assert [^\)]+\)'
        assertions = re.findall(pattern, assertions_str)
        # 解析并添加每个断言到求解器
        for assertion_str in assertions:
            # 解析断言并添加到求解器
            assertion = parse_smt2_string(assertion_str, decls={})
            for a in assertion:
                solver.add(a)
    return solver
