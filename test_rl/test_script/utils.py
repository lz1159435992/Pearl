import json
import sqlite3
from datetime import datetime

import time

from z3 import *
import math
import re

# import sys
#
# sys.path.append('/home/lz/PycharmProjects/Pearl')
import io
import json
import sys
from pysmt.walkers.identitydag import IdentityDagWalker
from pysmt.smtlib.parser import SmtLibParser
from pysmt.operators import ALL_TYPES, AND, OR, LE, LT, BV_ULT, BV_ULE, BV_SLT, BV_SLE, BV_COMP, EQUALS
import argparse
import numpy as np
from pysmt.exceptions import PysmtTypeError
import os
# from test_rl.test_script.utils import parse_smt2_in_parts, process_smt_lib_string, fetch_data_as_dict, \
#     solve_and_measure_time, model_to_dict, load_dictionary, extract_variables_from_smt2_content, normalize_variables
from collections import defaultdict

from pysmt.environment import get_env, push_env, Environment


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
            # print(line.strip())
            match = re.search(r'\(([^\)]+)\)\s*\)$', line.strip())
            if match:
                # 返回匹配到的最后一对括号内的内容
                # print(var_name,line,match)
                line_name = line.split(' ')[1]
                # print(line_name)
                if line_name == var_name:
                    return match.group(1)
               # return match.group(1)
    for line in lines:
        # 检查当前行是否包含变量声明
        if var_name in line and ("declare-fun" in line or "declare-const" in line):
            # print(line.strip())
            match = re.search(r'\(declare-fun\s+(\w+)\s*\(\)\s+(\w+)\s*\)', line.strip())
            if match:
                if match.group(1) == var_name:
                    return match.group(2)
            # return match.group(1)
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
    stats = solver.statistics()
    elapsed_time = stats.get_key_value('time')
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


def ast_contains_var(ast, var_name):
    """
    递归检查AST中是否包含给定的变量名。

    :param ast: 要检查的AST节点。
    :param var_name: 变量名字符串。
    :return: 布尔值，表示是否找到该变量名。
    """
    if ast.num_args() > 0:
        # 如果当前节点有子节点，递归检查每个子节点
        return any(ast_contains_var(arg, var_name) for arg in ast.children())
    elif ast.decl().kind() == Z3_OP_UNINTERPRETED:
        # 如果当前节点是一个叶节点且为未解释的符号（变量），检查其名字
        return str(ast) == var_name
    return False


def find_assertions_related_to_var_name(assertions, var_name):
    """
    找到与特定变量名相关的所有断言。

    :param assertions: Z3ast。
    :param var_name: 变量名字符串。
    :return: 包含指定变量名的所有断言列表。
    """
    related_assertions = []
    for assertion in assertions:
        if ast_contains_var(assertion, var_name):
            related_assertions.append(assertion)
    return related_assertions


def collect_symbols(ast, symbols):
    """
    递归收集AST中出现的所有符号。

    :param ast: 要检查的AST节点。
    :param symbols: 收集到的符号集合。
    """
    if ast.num_args() == 0:
        if ast.decl().kind() == Z3_OP_UNINTERPRETED:
            symbols.add(str(ast))
    else:
        for i in range(ast.num_args()):
            collect_symbols(ast.arg(i), symbols)


def find_assertions_related_to_var_names_optimized(assertions, var_names):
    """
    优化后的方法，找到与特定变量名列表中任何一个变量名相关的所有断言。

    :param solver: Z3求解器实例。
    :param var_names: 变量名字符串列表。
    :return: 一个字典，键为变量名，值为包含该变量名的所有断言列表。
    """
    related_assertions_dict = {var_name: [] for var_name in var_names}
    var_names_set = set(var_names)

    for assertion in assertions:
        symbols = set()
        collect_symbols(assertion, symbols)  # 收集当前断言中出现的所有符号

        # 检查收集到的符号中是否包含任何一个目标变量名
        if symbols.intersection(var_names_set):
            for var_name in var_names_set.intersection(symbols):
                related_assertions_dict[var_name].append(assertion)

    return related_assertions_dict


def dfs_ast_for_vars(ast, var_names, visited, results, var_nodes):
    """
    使用深度优先搜索（DFS）遍历AST，并检查是否包含给定的变量名列表中的任何一个变量名。

    :param ast: 要检查的AST节点。
    :param var_names: 变量名字符串列表。
    :param visited: 访问过的节点集合。
    :param results: 存储每个变量名是否被找到的字典。
    """
    stack = [ast]
    while stack:
        current_node = stack.pop()
        if id(current_node) in visited:
            continue
        visited.add(id(current_node))

        # 检查当前节点是否为未解释的符号（变量）
        if current_node.num_args() == 0 and current_node.decl().kind() == Z3_OP_UNINTERPRETED:
            var_name = str(current_node)
            if var_name in var_names:
                results[var_name] = True
                # 记录变量节点
                var_nodes[var_name].append(current_node)

        # 将子节点压入栈中
        for i in range(current_node.num_args()):
            stack.append(current_node.arg(i))

def dfs_ast_for_vars_range(ast, var_names, visited, results, var_nodes):
    """
    使用深度优先搜索（DFS）遍历AST，并检查是否包含给定的变量名列表中的任何一个变量名。

    :param ast: 要检查的AST节点。
    :param var_names: 变量名字符串列表。
    :param visited: 访问过的节点集合。
    :param results: 存储每个变量名是否被找到的字典。
    """
    stack = [ast]
    while stack:
        current_node = stack.pop()
        if id(current_node) in visited:
            continue
        visited.add(id(current_node))

        # 检查当前节点是否为未解释的符号（变量）
        # print(current_node.decl().kind())
        # print(current_node)
        print(f"Current node: {current_node}")
        print(f"Is app: {is_app(current_node)}")
        print(f"Is quantifier: {is_quantifier(current_node)}")
        if is_quantifier(current_node):
            body = current_node.body()  # Get the body of the quantifier
            # Recursively call your function to handle the body of the quantifier
            dfs_ast_for_vars_range(body, var_names, visited, results, var_nodes)
        elif is_app(current_node):
            if current_node.num_args() == 0 and current_node.decl().kind() == Z3_OP_UNINTERPRETED:
                var_name = str(current_node)
                if var_name in var_names:
                    results[var_name] = True
                    # 记录变量节点
                    var_nodes[var_name].append(current_node)

        # 将子节点压入栈中
        if is_app(current_node):
            for i in range(current_node.num_args()):
                stack.append(current_node.arg(i))
def find_assertions_related_to_var_names_optimized_dfs(assertions, var_names):
    """
    优化后的方法，找到与特定变量名列表中任何一个变量名相关的所有断言。

    :param solver: Z3求解器实例。
    :param var_names: 变量名字符串列表。
    :return: 一个字典，键为变量名，值为包含该变量名的所有断言列表。
    """
    results = {var_name: False for var_name in var_names}
    related_assertions_dict = {var_name: [] for var_name in var_names}

    visited = set()
    #记录一下z3变量
    var_nodes = {var_name: [] for var_name in var_names}

    var_range = {var_name: [] for var_name in var_names}
    for assertion in assertions:
        dfs_ast_for_vars(assertion, var_names, visited, results, var_nodes)
        for var_name in var_names:
            if results[var_name]:
                related_assertions_dict[var_name].append(assertion)
                # 查看assertion
                print(assertion)

    # 做一些简单的检查
    # for k, v in related_assertions_dict.items():
                s = Solver()
                opt = Optimize()
                s.add(assertion)
                # 创建求解器和优化器

                opt.add(s.assertions())

                # 设置优化目标
                opt.minimize(var_nodes[var_name][-1])

                # 检查优化结果
                result = opt.check()
                if result == sat:
                    print("Optimization result is satisfiable")
                    # 获取最小化后的变量的值
                    print("Minimum value of VAR1:", opt.model()[var_nodes[var_name][-1]])
                    min_value = opt.model()[var_nodes[var_name][-1]].as_long()
                else:
                    min_value = 0
                    print("Optimization result is not satisfiable")
                #获取最大值
                opt = Optimize()
                # s.add(assertion)
                # 创建求解器和优化器

                opt.add(s.assertions())

                # 设置优化目标
                opt.maximize(var_nodes[var_name][-1])

                # 检查优化结果
                result = opt.check()
                if result == sat:
                    print("Optimization result is satisfiable")
                    # 获取最大化后的变量的值
                    print("Maxmum value of VAR1:", opt.model()[var_nodes[var_name][-1]])
                    #只能处理正数的情况
                    max_value = opt.model()[var_nodes[var_name][-1]].as_long()
                else:
                    print("Optimization result is not satisfiable")
                var_range[var_name].append([min_value, max_value])

        # 重置results，以便下一次断言检查
        results = {var_name: False for var_name in var_names}
    return related_assertions_dict, var_range
#新修改的
def solve_assertion_get_range(assertions, var_names):
    """
    优化后的方法，找到与特定变量名列表中任何一个变量名相关的所有断言。

    :param solver: Z3求解器实例。
    :param var_names: 变量名字符串列表。
    :return: 一个字典，键为变量名，值为包含该变量名的所有断言列表。
    """
    results = {var_name: False for var_name in var_names}
    related_assertions_dict = {var_name: [] for var_name in var_names}


    #记录一下z3变量
    var_nodes = {var_name: [] for var_name in var_names}

    var_range = {var_name: [] for var_name in var_names}
    for assertion in assertions:
        print('*******************')
        print(assertion)
        visited = set()
        dfs_ast_for_vars_range(assertion, var_names, visited, results, var_nodes)
        for var_name in var_names:
            if results[var_name]:
                related_assertions_dict[var_name].append(assertion)
                # 查看assertion
                # print(assertion)

    # 做一些简单的检查
    # for k, v in related_assertions_dict.items():
    #             print(var_name)
    #             print('-----------------------------------')
    #             s = Solver()
    #             opt = Optimize()
    #             s.add(assertion)
    #             # 创建求解器和优化器
    #
    #             opt.add(s.assertions())
    #
    #             # 设置优化目标
    #             opt.minimize(var_nodes[var_name][-1])
    #
    #             # 检查优化结果
    #             result = opt.check()
    #             if result == sat:
    #                 print("Optimization result is satisfiable")
    #                 # 获取最小化后的变量的值
    #                 print("Minimum value:", opt.model()[var_nodes[var_name][-1]])
    #                 min_value = opt.model()[var_nodes[var_name][-1]].as_long()
    #             else:
    #                 min_value = 0
    #                 print("Optimization result is not satisfiable")
    #             #获取最大值
    #             opt = Optimize()
    #             # s.add(assertion)
    #             # 创建求解器和优化器
    #
    #             opt.add(s.assertions())
    #
    #             # 设置优化目标
    #             opt.maximize(var_nodes[var_name][-1])
    #
    #             # 检查优化结果
    #             result = opt.check()
    #             if result == sat:
    #                 print("Optimization result is satisfiable")
    #                 # 获取最大化后的变量的值
    #                 print("Maxmum value of VAR1:", opt.model()[var_nodes[var_name][-1]])
    #                 #只能处理正数的情况
    #                 max_value = opt.model()[var_nodes[var_name][-1]].as_long()
    #             else:
    #                 print("Optimization result is not satisfiable")
    #             var_range[var_name].append([min_value, max_value])
    #             print('-----------------------------------')


        # 重置results，以便下一次断言检查
        results = {var_name: False for var_name in var_names}
    return related_assertions_dict, var_range

def extract_variables_from_smt2_content(content):
    """
    从 SMT2 格式的字符串内容中提取变量名，排除布尔类型的变量。

    参数:
    - content: SMT2 格式的字符串内容。

    返回:
    - 非布尔类型变量名列表。
    """
    # 用于匹配 `(declare-fun ...)` 语句的正则表达式，包括变量名和类型
    variable_pattern = re.compile(r'\(declare-fun\s+([^ ]+)\s*\(\s*\)\s*([^)]+)\)')

    # 存储提取的非布尔类型变量名
    variables = []

    # 按行分割字符串并迭代每一行
    for line in content.splitlines():
        # 在每一行中查找匹配的变量声明
        match = variable_pattern.search(line)
        if match:
            var_name, var_type = match.group(1, 2)
            # 如果变量类型不是 Bool，则将变量名添加到列表中
            if var_type != 'Bool':
                variables.append(var_name.replace('|', ''))

    return variables


# 保存长文本
def save_string_to_file(file_path, new_string):
    if not os.path.exists(file_path):
        # 文件不存在时，创建文件
        strings = []
        with open(file_path, 'w') as file:
            json.dump(strings, file, indent=4)
        # print(f"文件 info_dict_bingxing.txt 已创建。")
    else:
        with open(file_path, 'r') as file:
            strings = json.load(file)
    strings.append(new_string)

    with open(file_path, 'w') as file:
        json.dump(strings, file)


def repalce_veriable(input_string, variable_pred, selected_int, type_scale):
    # 定义替换的正则表达式
    replacement_pattern = r"(assert (= {} (_ bv{} {})))".format(
        variable_pred, selected_int, type_scale
    )

    # 使用正则表达式替换字符串中的变量
    output_string = re.sub(r"\(assert \(= {} \(_ bv(\w+) (\w+)\)\)\)".format(variable_pred), replacement_pattern,
                           input_string)

    # 打印输出结果
    # print(output_string)
    return output_string


# 标准化方法

def generate_variable_names(count):
    """
    根据固定的规则生成变量名列表。

    :param count: 要生成的变量名的数量。
    :return: 一个包含生成的变量名的列表。
    """
    return [f"VAR{i + 1}" for i in range(count)]


def create_variables_dict(variable_names, variable_values):
    """
    根据变量名列表和变量值列表创建一个字典。

    :param variable_names: 一个包含变量名的列表。
    :param variable_values: 一个包含对应替换值的列表。
    :return: 一个字典，其键是变量名，值是对应的替换值。
    """
    if len(variable_names) != len(variable_values):
        raise ValueError("变量名列表和变量值列表的长度必须相同。")

    return dict(zip(variable_names, variable_values))


def normalize_variables(text, variable_values):
    """
    替换文本中的变量为指定的值。
    key：原来的变量名
    value：替换的值

    :param text: 原始文本，包含需要替换的变量。
    :param variables: 一个包含变量名和对应替换值的字典。
    :return: 替换后的文本。
    """
    # variable_count = 3  # 想要生成的变量数量
    variable_names = generate_variable_names(len(variable_values))

    variables_dict = create_variables_dict(variable_values, variable_names)
    # print(variables_dict)
    # 按照变量列表的顺序进行替换
    for var_name, var_value in variables_dict.items():
        # 使用正则表达式替换变量，确保只替换完整的单词，避免替换中间包含变量名的单词
        text = re.sub(r'\b' + re.escape(var_name) + r'\b', str(var_value), text)
    return text


# 根据smt文件中的变量的各种属性生成变量顺序进行替换
class ASTBuilder(IdentityDagWalker):

    def __init__(self, env=None, invalidate_memoization=None):
        super().__init__(env, invalidate_memoization)
        self.nodeCounter = None
        self.id_to_counter = None
        self.edges = None
        self.edge_attr = None
        self.symbol_to_id = None
        self.constant_to_id = None
        self.variable_clause_count = None
        self.variable_constant_clause_count = None
        self.variable_type = None
        self.variable_frequency = None
        self.variable_logic_operation_count = None  # Store counts of logic operations
        self.variable_clause_size = None
        self.constant_list = []  # New data structure for storing constants
        self.variable_bounds = defaultdict(lambda: {'lower': [], 'upper': [], 'equal': [], 'low_var': [], 'up_var': [],
                                                    'equal_var': []})  # New data structure for bounds

    def walk(self, formula, **kwargs):
        if formula in self.memoization:
            return self.memoization[formula]

        self.nodeCounter = 0
        self.nodes = []
        self.edges = [[], []]
        self.edge_attr = []
        self.id_to_counter = dict()
        self.symbol_to_node = dict()
        self.constant_to_node = dict()
        self.variable_clause_count = dict()
        self.variable_constant_clause_count = dict()
        self.variable_type = dict()
        self.variable_frequency = dict()
        self.variable_logic_operation_count = defaultdict(int)
        self.variable_clause_size = dict()
        res = self.iter_walk(formula, **kwargs)

        if self.invalidate_memoization:
            self.memoization.clear()

        return res

    def add_node(self, formula):
        node_rep = [0] * (len(ALL_TYPES) + 1)
        node_rep[formula.node_type()] = 1
        self.nodes.append(node_rep)
        assert self.nodeCounter == len(self.nodes)

    def get_node_counter(self, formula, parent):
        value = None
        if formula.is_symbol() and not parent:
            if formula.node_id() in self.symbol_to_node:
                self.symbol_to_node[formula.node_id()].append(self.nodeCounter)
            else:
                self.symbol_to_node[formula.node_id()] = [self.nodeCounter]
            self.id_to_counter[formula.node_id()] = self.nodeCounter
            value = self.nodeCounter

            self.nodeCounter += 1
            self.add_node(formula)

            # Collect variable type
            self.variable_type[formula] = formula.symbol_type()

            # Collect variable frequency
            if formula not in self.variable_frequency:
                self.variable_frequency[formula] = 0
            self.variable_frequency[formula] += 1

        elif formula.is_constant() and not parent:
            if formula.node_id() in self.symbol_to_node:
                self.constant_to_node[formula.node_id()].append(self.nodeCounter)
            else:
                self.constant_to_node[formula.node_id()] = [self.nodeCounter]

            self.id_to_counter[formula.node_id()] = self.nodeCounter
            value = self.nodeCounter

            self.nodeCounter += 1
            self.add_node(formula)

            # Add constant to the list
            if formula not in self.constant_list:
                self.constant_list.append(formula)  # Collect constants

        elif formula.node_id() not in self.id_to_counter:
            self.id_to_counter[formula.node_id()] = self.nodeCounter

            value = self.nodeCounter
            self.nodeCounter += 1
            self.add_node(formula)
        else:
            value = self.id_to_counter[formula.node_id()]

        return value

    def _push_with_children_to_stack(self, formula, **kwargs):
        """Add children to the stack."""
        self.stack.append((True, formula))

        parenId = self.get_node_counter(formula, True)

        for s in self._get_children(formula):
            # Add only if not memoized already
            childId = self.get_node_counter(s, False)
            self.edges[0].append(parenId)
            self.edges[1].append(childId)
            self.edge_attr.append(0)

            self.edges[0].append(childId)
            self.edges[1].append(parenId)
            self.edge_attr.append(1)

            if s.is_symbol():
                if s not in self.variable_clause_count:
                    self.variable_clause_count[s] = set()
                self.variable_clause_count[s].add(parenId)

                # Check if this clause also involves a constant
                for sub_s in self._get_children(formula):
                    if sub_s.is_constant():
                        if s not in self.variable_constant_clause_count:
                            self.variable_constant_clause_count[s] = set()
                        self.variable_constant_clause_count[s].add(parenId)
                        break

                # Collect logic operations
                self.variable_logic_operation_count[s] += 1  # Count the logic operation

                # Collect clause size
                self.variable_clause_size[s] = len(self._get_children(formula))

            key = self._get_key(s, **kwargs)
            if key not in self.memoization:
                self.stack.append((False, s))
            # 一个约束只判断一次
        # # 一个约束只判断一次  去掉这部分内容
        # if len(formula.args()) == 2:
        #     left, right = formula.arg(0), formula.arg(1)
        #     # 获取变量间的大小关系
        #     print(left, left.node_type(), right, right.node_type(), formula.node_type())
        #     # if sub_s.is_constant():
        #     #     constant_value = sub_s.constant_value()
        #     if formula.node_type() in [LE, LT, BV_ULT, BV_ULE, BV_ULT, BV_SLT, BV_SLE]:
        #         # 左边变量小于常量
        #         if left.is_symbol() and right.is_constant():
        #             constant_value = right.constant_value()
        #             self.variable_bounds[left]['upper'].append(constant_value)
        #         # 右边变量小于常量
        #         elif right.is_symbol() and left.is_constant():
        #             constant_value = left.constant_value()
        #             self.variable_bounds[right]['lower'].append(constant_value)
        #         # 左边变量小于右边变量
        #         elif left.is_symbol() and right.is_symbol():
        #             self.variable_bounds[left]['up_var'].append(right)
        #         # 考虑存在线性约束的情况  三个类型 PLUS, MINUS, TIMES  之后添加
        #     elif formula.node_type() in [BV_COMP, EQUALS]:
        #         if left.is_symbol() and right.is_constant():
        #             constant_value = right.constant_value()
        #             self.variable_bounds[left]['equal'].append(constant_value)
        #         elif right.is_symbol() and left.is_constant():
        #             constant_value = left.constant_value()
        #             self.variable_bounds[right]['equal'].append(constant_value)
        #         else:
        #             self.variable_bounds[left]['equal_var'].append(right)


# 根据smt文件中的变量的各种属性生成变量顺序进行替换，返回生成后的smtlib_str
def normalize_smt_str(smtlib_str):
    with Environment() as env:
        file_obj = io.StringIO(smtlib_str)
        # try:
        myParser = SmtLibParser(env)
        formula = None
        try:
            formula = myParser.get_script(file_obj).get_last_formula()
            file_obj.close()

            astBuilder = ASTBuilder()
            astBuilder.walk(formula)
            assert len(astBuilder.edges[0]) == len(astBuilder.edges[1])

            nodes = astBuilder.nodes
            edges = astBuilder.edges
            edge_attr = astBuilder.edge_attr

            for symbol in astBuilder.symbol_to_node.values():
                if len(symbol) < 2:
                    continue
                repr = [0] * (len(ALL_TYPES) + 1)
                repr[-1] = 1
                nodes.append(repr)
                for node in symbol:
                    # TO Uber symbol node
                    edges[0].append(node)
                    edges[1].append(len(nodes) - 1)
                    edge_attr.append(2)

            nodes = np.array(nodes)
            edges = np.array(edges)
            edge_attr = np.array(edge_attr)

            assert sum(edge_attr == 0) == sum(edge_attr == 1)

            # Output variable attributes
            variable_clause_count = {str(var): len(clauses) for var, clauses in
                                     astBuilder.variable_clause_count.items()}
            variable_constant_clause_count = {str(var): len(clauses) for var, clauses in
                                              astBuilder.variable_constant_clause_count.items()}
            variable_type = {str(var): str(var_type) for var, var_type in astBuilder.variable_type.items()}
            variable_frequency = {str(var): freq for var, freq in astBuilder.variable_frequency.items()}
            variable_logic_operation_count = {str(var): count for var, count in
                                              astBuilder.variable_logic_operation_count.items()}
            variable_clause_size = {str(var): size for var, size in astBuilder.variable_clause_size.items()}
            # # 添加对变量边界的判断
            # variable_bounds = {str(var): bounds for var, bounds in astBuilder.variable_bounds.items()}

            # Combine attributes for sorting
            combined_attributes = {var: (
                variable_clause_size.get(var, 0),
                variable_clause_count.get(var, 0),
                variable_frequency.get(var, 0),
                variable_logic_operation_count.get(var, 0),  # Total logic operations count
                variable_constant_clause_count.get(var, 0),
            ) for var in variable_clause_count.keys()}
            # 也可以直接提取变量名列表 使用extract_variables_from_smt2_content(smtlib_str)

            # Sort variables based on combined attributes
            sorted_variables = sorted(combined_attributes.keys(), key=lambda x: (
                combined_attributes[x][0],  # Variable Clause Sizes
                combined_attributes[x][1],  # Variable to Clause Count
                combined_attributes[x][2],  # Variable Frequencies
                combined_attributes[x][3],  # Total Logic Operations Count
                combined_attributes[x][4],  # Variable Constant Clause Count
            ), reverse=True)

            # Generate a dictionary with sorted variables and their new names
            sorted_variable_dict = {var: f"VAR{i + 1}" for i, var in enumerate(sorted_variables)}

            # except PysmtTypeError as e:
            #     print("未知错误：", e)
            # variable_bounds = {var: variable_bounds[var] for var in sorted_variables if var in variable_bounds.keys()}
            for var_name, var_value in sorted_variable_dict.items():
                # 使用正则表达式替换变量，确保只替换完整的单词，避免替换中间包含变量名的单词
                smtlib_str = re.sub(r'\b' + re.escape(var_name) + r'\b', str(var_value), smtlib_str)

            # Extract constants and append to the result
            constant_list = [str(constant) for constant in astBuilder.constant_list]
            constant_list = sorted(constant_list)
            print(constant_list)
            # 对常量值进行处理，只获取具体值
            constants_set = set()

            for const in constant_list:
                if const == 'True' or const == 'False':
                    continue
                if '_' in const:
                    value, width = const.split('_')
                    constants_set.add(int(value))
            # 去掉连续的常量值
            constants = sorted(list(constants_set))
            print('排序后列表')
            print(constants)
            filtered_constants = set()
            last_num = None
            for num in constants:
                if last_num is not None and num == last_num + 1:
                    last_num = num
                    continue
                filtered_constants.add(num)
                last_num = num
        except PysmtTypeError as e:
            print("未知错误：", e)
            return smtlib_str, None, None

    return smtlib_str, sorted_variable_dict, list(filtered_constants)

    # # Remove consecutive natural numbers if needed
    # constants = sorted(constants)
    # filtered_constants = set()
    # last_num = None
    # for num in constants:
    #     if last_num is not None and num == last_num + 1:
    #         last_num = num
    #         continue
    #     filtered_constants.add(num)
    #     last_num = num


# 只用来获取拥有顺序的变量列表
def normalize_smt_str_without_replace(smtlib_str):
    try:
        with Environment() as env:
            file_obj = io.StringIO(smtlib_str)
            # try:
            myParser = None
            myParser = SmtLibParser(env)
            formula = None
            formula = myParser.get_script(file_obj).get_last_formula()
            file_obj.close()

            astBuilder = ASTBuilder()
            astBuilder.walk(formula)
            assert len(astBuilder.edges[0]) == len(astBuilder.edges[1])

            nodes = astBuilder.nodes
            edges = astBuilder.edges
            edge_attr = astBuilder.edge_attr

            for symbol in astBuilder.symbol_to_node.values():
                if len(symbol) < 2:
                    continue
                repr = [0] * (len(ALL_TYPES) + 1)
                repr[-1] = 1
                nodes.append(repr)
                for node in symbol:
                    # TO Uber symbol node
                    edges[0].append(node)
                    edges[1].append(len(nodes) - 1)
                    edge_attr.append(2)

            nodes = np.array(nodes)
            edges = np.array(edges)
            edge_attr = np.array(edge_attr)

            assert sum(edge_attr == 0) == sum(edge_attr == 1)

            # Output variable attributes
            variable_clause_count = {str(var): len(clauses) for var, clauses in
                                     astBuilder.variable_clause_count.items()}
            variable_constant_clause_count = {str(var): len(clauses) for var, clauses in
                                              astBuilder.variable_constant_clause_count.items()}
            variable_type = {str(var): str(var_type) for var, var_type in astBuilder.variable_type.items()}
            variable_frequency = {str(var): freq for var, freq in astBuilder.variable_frequency.items()}
            variable_logic_operation_count = {str(var): count for var, count in
                                              astBuilder.variable_logic_operation_count.items()}
            variable_clause_size = {str(var): size for var, size in astBuilder.variable_clause_size.items()}

            # Combine attributes for sorting
            combined_attributes = {var: (
                variable_clause_size.get(var, 0),
                variable_clause_count.get(var, 0),
                variable_frequency.get(var, 0),
                variable_logic_operation_count.get(var, 0),  # Total logic operations count
                variable_constant_clause_count.get(var, 0),
            ) for var in variable_clause_count.keys()}
            # 也可以直接提取变量名列表 使用extract_variables_from_smt2_content(smtlib_str)

            # Sort variables based on combined attributes
            sorted_variables = sorted(combined_attributes.keys(), key=lambda x: (
                combined_attributes[x][0],  # Variable Clause Sizes
                combined_attributes[x][1],  # Variable to Clause Count
                combined_attributes[x][2],  # Variable Frequencies
                combined_attributes[x][3],  # Total Logic Operations Count
                combined_attributes[x][4],  # Variable Constant Clause Count
            ), reverse=True)

            # except PysmtTypeError as e:
            #     print("未知错误：", e)
            # Extract constants and append to the result
            sorted_variables = [var for var in sorted_variables if variable_type[var] != 'Bool']
    except Exception as e:
        print("未知错误：", e)
        sorted_variables = extract_variables_from_smt2_content(smtlib_str)
    return sorted_variables
