import json
import sqlite3
from datetime import datetime

import time

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


def find_var_declaration_in_string(smt_content, var_name):
    """
    在 SMT-LIB 内容字符串中查找并返回给定变量名的声明类型。
    """
    # 将字符串按行分割成列表进行处理
    lines = smt_content.split('\n')

    for line in lines:
        # 检查当前行是否包含变量声明
        if var_name in line and ("declare-fun" in line or "declare-const" in line):
            print(line.strip())
            match = re.search(r'\(([^\)]+)\)\s*\)$', line.strip())
            if match:
                # 返回匹配到的最后一对括号内的内容
                return match.group(1)
            else:
                return None
    return None


def split_at_check_sat(smt_string):
    # 查找第一个出现的 (check-sat) 指令
    pos = smt_string.find('(check-sat)')

    # 如果找到了 (check-sat)，则进行切分
    if pos != -1:
        # 切分为两部分：(check-sat) 之前和之后的部分
        before_check_sat = smt_string[:pos]
        after_check_sat = smt_string[pos:]

        return before_check_sat, after_check_sat
    else:
        # 如果没有找到 (check-sat)，返回原字符串和空字符串
        return smt_string, ''


def extract_variables_from_smt2_content(content):
    """
    从 SMT2 格式的字符串内容中提取变量名。

    参数:
    - content: SMT2 格式的字符串内容。

    返回:
    - 变量名列表。
    """
    # 用于匹配 `(declare-fun ...)` 语句中的变量名的正则表达式
    variable_pattern = re.compile(r'\(declare-fun\s+([^ ]+)')

    # 存储提取的变量名
    variables = []

    # 按行分割字符串并迭代每一行
    for line in content.splitlines():
        # 在每一行中查找匹配的变量名
        match = variable_pattern.search(line)
        if match:
            # 如果找到匹配项，则将变量名添加到列表中
            variables.append(match.group(1).replace('|', ''))

    return set(variables)


# 将数据库中的内容保存为字典
def fetch_data_as_dict(db_path, table_name):
    # 连接到SQLite数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 查询表中的所有键值对
    query = f"SELECT key, value FROM {table_name}"
    cursor.execute(query)
    rows = cursor.fetchall()

    # 关闭数据库连接
    conn.close()

    # 将查询结果转换为字典
    result_dict = {row[0]: row[1] for row in rows}
    return result_dict


def solve_and_measure_time(solver, timeout):
    solver.set("timeout", timeout)
    start_time = time.time()
    result = solver.check()
    # print(result)
    elapsed_time = time.time() - start_time
    if result == sat:
        return "sat", solver.model(), elapsed_time
    elif result == unknown:
        return "unknown", None, elapsed_time
    else:
        return "unsat", None, elapsed_time


def update_txt_with_info(file_path, info):
    # current_time = time.time()
    with open(file_path, "a") as file:
        file.write(f"info:{info}\n")


# 读取json文件为字典
def load_dictionary(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)