import json
import random
import re
from z3 import *
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from z3 import *
# def dfs_ast_for_vars(ast, var_names, visited, related_assertions_dict):
#     """
#     使用深度优先搜索（DFS）遍历AST，并检查是否包含给定的变量名列表中的任何一个变量名。
#     同时，将包含任一变量的断言添加到相关的断言列表中。
#
#     :param ast: 要检查的AST节点。
#     :param var_names: 变量名字符串列表。
#     :param visited: 访问过的节点集合。
#     :param related_assertions_dict: 存储每个变量名相关的断言列表的字典。
#     """
#     stack = [ast]
#     while stack:
#         current_node = stack.pop()
#         if id(current_node) in visited:
#             continue
#         visited.add(id(current_node))
#
#         # 检查当前节点是否为未解释的符号（变量）
#         if current_node.num_args() == 0 and current_node.decl().kind() == Z3_OP_UNINTERPRETED:
#             var_name = str(current_node)
#             if var_name in var_names:
#                 for related_var_name in var_names:
#                     related_assertions_dict[related_var_name].append(current_node)
#
#         # 将子节点压入栈中
#         for i in range(current_node.num_args()):
#             stack.append(current_node.arg(i))
#
# def find_assertions_related_to_var_names_optimized_dfs(solver, var_names):
#     """
#     优化后的方法，找到与特定变量名列表中任何一个变量名相关的所有断言。
#
#     :param solver: Z3求解器实例。
#     :param var_names: 变量名字符串列表。
#     :return: 一个字典，键为变量名，值为包含该变量名的所有断言列表。
#     """
#     related_assertions_dict = {var_name: [] for var_name in var_names}
#     visited = set()
#
#     for assertion in solver.assertions():
#         dfs_ast_for_vars(assertion, var_names, visited, related_assertions_dict)
#
#     return related_assertions_dict


def dfs_ast_for_vars(ast, var_names, visited, results):
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

        # 将子节点压入栈中
        for i in range(current_node.num_args()):
            stack.append(current_node.arg(i))

def find_assertions_related_to_var_names_optimized_dfs(solver, var_names):
    """
    优化后的方法，找到与特定变量名列表中任何一个变量名相关的所有断言。

    :param solver: Z3求解器实例。
    :param var_names: 变量名字符串列表。
    :return: 一个字典，键为变量名，值为包含该变量名的所有断言列表。
    """
    results = {var_name: False for var_name in var_names}
    related_assertions_dict = {var_name: [] for var_name in var_names}
    visited = set()

    for assertion in solver.assertions():
        dfs_ast_for_vars(assertion, var_names, visited, results)
        for var_name in var_names:
            if results[var_name]:
                related_assertions_dict[var_name].append(assertion)
        # 重置results，以便下一次断言检查
        results = {var_name: False for var_name in var_names}

    return related_assertions_dict


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


# def find_assertions_related_to_var_names_optimized(solver, var_names):
#     """
#     优化后的方法，找到与特定变量名列表中任何一个变量名相关的所有断言。
#
#     :param solver: Z3求解器实例。
#     :param var_names: 变量名字符串列表。
#     :return: 一个字典，键为变量名，值为包含该变量名的所有断言列表。
#     """
#     related_assertions_dict = {var_name: [] for var_name in var_names}
#     var_names_set = set(var_names)
#
#     # 使用栈来模拟深度优先遍历
#     stack = list(solver.assertions())
#
#     while stack:
#         current = stack.pop()
#         symbols = set()
#         collect_symbols(current, symbols)  # 收集当前断言中出现的所有符号
#
#         # 检查收集到的符号中是否包含任何一个目标变量名
#         if symbols.intersection(var_names_set):
#             for var_name in var_names_set.intersection(symbols):
#                 related_assertions_dict[var_name].append(current)
#
#         # 将当前节点的子节点加入栈中
#         for arg in current.children():
#             stack.append(arg)
#
#     return related_assertions_dict


def find_assertions_related_to_var_names_optimized(solver, var_names):
    """
    优化后的方法，找到与特定变量名列表中任何一个变量名相关的所有断言。

    :param solver: Z3求解器实例。
    :param var_names: 变量名字符串列表。
    :return: 一个字典，键为变量名，值为包含该变量名的所有断言列表。
    """
    related_assertions_dict = {var_name: [] for var_name in var_names}
    var_names_set = set(var_names)

    for assertion in solver.assertions():
        symbols = set()
        collect_symbols(assertion, symbols)  # 收集当前断言中出现的所有符号

        # 检查收集到的符号中是否包含任何一个目标变量名
        if symbols.intersection(var_names_set):
            for var_name in var_names_set.intersection(symbols):
                related_assertions_dict[var_name].append(assertion)

    return related_assertions_dict


def collect_vars(ast, collected_vars):
    """
    递归收集AST中出现的所有变量名。

    :param ast: 要检查的AST节点。
    :param collected_vars: 已收集变量的集合。
    """
    if ast.num_args() == 0:  # 叶节点
        if ast.decl().kind() == Z3_OP_UNINTERPRETED:
            # 将变量名添加到集合中
            collected_vars.add(str(ast))
    else:
        # 遍历所有子节点
        for i in range(ast.num_args()):
            collect_vars(ast.arg(i), collected_vars)


def find_assertions_related_to_vars(solver, var_names):
    """
    找到与特定变量名列表中任何一个变量名相关的所有断言。

    :param solver: Z3求解器实例。
    :param var_names: 变量名字符串列表。
    :return: 一个字典，键为变量名，值为包含该变量名的所有断言列表。
    """
    related_assertions_dict = {var_name: [] for var_name in var_names}
    var_names_set = set(var_names)

    for assertion in solver.assertions():
        collected_vars = set()
        collect_vars(assertion, collected_vars)
        # 如果收集到的变量中有目标变量名，记录该断言
        if not collected_vars.isdisjoint(var_names_set):
            intersected_vars = collected_vars.intersection(var_names_set)
            for var in intersected_vars:
                related_assertions_dict[var].append(assertion)

    return related_assertions_dict


def dfs_ast_for_vars(ast, var_names, visited, results):
    """
    使用深度优先搜索（DFS）遍历AST，并检查是否包含给定的变量名列表中的任何一个变量名。

    :param ast: 要检查的AST节点。
    :param var_names: 变量名字符串列表。
    :param visited: 访问过的节点集合。
    :param results: 存储每个变量名是否被找到的字典。
    """
    # 如果已经访问过该节点，则返回
    if id(ast) in visited:
        return
    visited.add(id(ast))

    # 将当前节点转换成字符串
    ast_str = str(ast)

    # 检查当前节点的字符串表示中是否包含任何一个目标变量名
    for var_name in var_names:
        if var_name in ast_str:
            results[var_name] = True

    # 遍历所有子节点
    for i in range(ast.num_args()):
        dfs_ast_for_vars(ast.arg(i), var_names, visited, results)


def find_assertions_related_to_var_names_optimized(solver, var_names):
    """
    优化后的方法，找到与特定变量名列表中任何一个变量名相关的所有断言。

    :param solver: Z3求解器实例。
    :param var_names: 变量名字符串列表。
    :return: 一个字典，键为变量名，值为包含该变量名的所有断言列表。
    """
    related_assertions_dict = {var_name: [] for var_name in var_names}
    visited = set()

    for assertion in solver.assertions():
        results = {var_name: False for var_name in var_names}
        dfs_ast_for_vars(assertion, var_names, visited, results)
        for var_name, found in results.items():
            if found:
                related_assertions_dict[var_name].append(assertion)

    return related_assertions_dict


def ast_contains_var_memo(ast, var_names, memo):
    """
    递归检查AST中是否包含给定的变量名列表中的任何一个变量名，使用记忆化以提高性能。

    :param ast: 要检查的AST节点。
    :param var_names: 变量名字符串列表。
    :param memo: 用于记忆化的字典。
    :return: 布尔值，表示是否找到列表中的任一变量名。
    """
    # 将当前节点转化为一个唯一标识符，例如，其字符串表示形式
    key = (str(ast), tuple(sorted(var_names)))
    if key in memo:
        return memo[key]

    if ast.num_args() > 0:
        # 如果当前节点有子节点，递归检查每个子节点
        result = any(ast_contains_var_memo(arg, var_names, memo) for arg in ast.children())
    elif ast.decl().kind() == Z3_OP_UNINTERPRETED:
        # 如果当前节点是一个叶节点且为未解释的符号（变量），检查其名字是否在列表中
        result = str(ast) in var_names
    else:
        result = False

    memo[key] = result
    return result


def find_assertions_related_to_var_names_memo(solver, var_names):
    """
    找到与特定变量名列表中任何一个变量名相关的所有断言，使用记忆化以提高性能。

    :param solver: Z3求解器实例。
    :param var_names: 变量名字符串列表。
    :return: 一个字典，键为变量名，值为包含该变量名的所有断言列表。
    """
    memo = {}
    related_assertions_dict = {var_name: [] for var_name in var_names}
    for assertion in solver.assertions():
        for var_name in var_names:
            # 使用记忆化版本的函数
            if ast_contains_var_memo(assertion, [var_name], memo):
                related_assertions_dict[var_name].append(assertion)
    return related_assertions_dict


def ast_contains_var(ast, var_names):
    """
    递归检查AST中是否包含给定的变量名列表中的任何一个变量名。

    :param ast: 要检查的AST节点。
    :param var_names: 变量名字符串列表。
    :return: 布尔值，表示是否找到列表中的任一变量名。
    """
    if ast.num_args() > 0:
        # 如果当前节点有子节点，递归检查每个子节点
        return any(ast_contains_var(arg, var_names) for arg in ast.children())
    elif ast.decl().kind() == Z3_OP_UNINTERPRETED:
        # 如果当前节点是一个叶节点且为未解释的符号（变量），检查其名字是否在列表中
        return str(ast) in var_names
    return False


def find_assertions_related_to_var_names(solver, var_names):
    """
    找到与特定变量名列表中任何一个变量名相关的所有断言。

    :param solver: Z3求解器实例。
    :param var_names: 变量名字符串列表。
    :return: 一个字典，键为变量名，值为包含该变量名的所有断言列表。
    """
    related_assertions_dict = {var_name: [] for var_name in var_names}
    for assertion in solver.assertions():
        for var_name in var_names:
            # 检查当前断言是否与列表中的某个变量名相关
            if ast_contains_var(assertion, [var_name]):
                related_assertions_dict[var_name].append(assertion)
    return related_assertions_dict


# def ast_contains_var(ast, var_name):
#     """
#     递归检查AST中是否包含给定的变量名。
#
#     :param ast: 要检查的AST节点。
#     :param var_name: 变量名字符串。
#     :return: 布尔值，表示是否找到该变量名。
#     """
#     if ast.num_args() > 0:
#         # 如果当前节点有子节点，递归检查每个子节点
#         with ThreadPoolExecutor() as executor:
#             futures = [executor.submit(ast_contains_var, arg, var_name) for arg in ast.children()]
#             return any(future.result() for future in as_completed(futures))
#     elif ast.decl().kind() == Z3_OP_UNINTERPRETED:
#         # 如果当前节点是一个叶节点且为未解释的符号（变量），检查其名字
#         return str(ast) == var_name
#     return False
#
# def find_assertions_related_to_var_name(solver, var_name):
#     """
#     找到与特定变量名相关的所有断言（多线程版本）。
#
#     :param solver: Z3求解器实例。
#     :param var_name: 变量名字符串。
#     :return: 包含指定变量名的所有断言列表。
#     """
#     related_assertions = []
#
#     def check_assertion(assertion):
#         if ast_contains_var(assertion, var_name):
#             related_assertions.append(assertion)
#
#     with ThreadPoolExecutor() as executor:
#         # 提交所有断言到线程池进行检查
#         executor.map(check_assertion, solver.assertions())
#
#     return related_assertions

# def ast_contains_var(ast, var_name):
#     """
#     递归检查AST中是否包含给定的变量名。
#
#     :param ast: 要检查的AST节点。
#     :param var_name: 变量名字符串。
#     :return: 布尔值，表示是否找到该变量名。
#     """
#     if ast.num_args() > 0:
#         # 如果当前节点有子节点，递归检查每个子节点
#         return any(ast_contains_var(arg, var_name) for arg in ast.children())
#     elif ast.decl().kind() == Z3_OP_UNINTERPRETED:
#         # 如果当前节点是一个叶节点且为未解释的符号（变量），检查其名字
#         return str(ast) == var_name
#     return False


def find_assertions_related_to_var_name(solver, var_name):
    """
    找到与特定变量名相关的所有断言。

    :param solver: Z3求解器实例。
    :param var_name: 变量名字符串。
    :return: 包含指定变量名的所有断言列表。
    """
    related_assertions = []
    for assertion in solver.assertions():
        if ast_contains_var(assertion, var_name):
            print(assertion)
            related_assertions.append(assertion)
    return related_assertions


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


print(time.time())
file_path = '/home/lz/baidudisk/smt/gnu_angr.tar.gz/single_test/who/who86404'
with open('time.txt', "a") as file:
    file.write(f"当前测试文件:{file_path}\n")
with open(file_path, 'r') as file:
    # 读取文件所有内容到一个字符串
    smtlib_str = file.read()
# 解析字符串
try:
    # 将JSON字符串转换为字典
    dict_obj = json.loads(smtlib_str)
    # print("转换后的字典：", dict_obj)
except json.JSONDecodeError as e:
    print("解析错误：", e)
#
if 'smt-comp' in file_path:
    smtlib_str = dict_obj['smt_script']
else:
    smtlib_str = dict_obj['script']
smtlib_str_before, smtlib_str_after = split_at_check_sat(smtlib_str)
# print(smtlib_str_before, '************************************', smtlib_str_after)

# Extract variables from each assertion
# for a in assertions:
#     visit(a)
variables = extract_variables_from_smt2_content(smtlib_str)

# Print all variables
print("变量列表：")
for v in variables:
    print(v)
    variable_pred = v

selected_int = 455674

type_info = find_var_declaration_in_string(smtlib_str, variable_pred)
print(type_info)
print(type(type_info))
# if type_info == '_ BitVec 64':
#     new_constraint = "(assert (= {} (_ bv{} 64)))\n".format(variable_pred,selected_int)
# elif type_info == '_ BitVec 8':
#     new_constraint = "(assert (= {} (_ bv{} 8)))\n".format(variable_pred,selected_int)
# elif type_info == '_ BitVec 1008':
#     new_constraint = "(assert (= {} (_ bv{} 1008)))\n".format(variable_pred,selected_int)
# smtlib_str = smtlib_str_before + new_constraint + smtlib_str_after
assertions = parse_smt2_string(smtlib_str)
# assertions = process_smt_lib_string(smtlib_str)
# variables = set()

solver = Solver()
assertions_list = []
for a in assertions:
    # print(a)
    solver.add(a)
#     for i in a:
#         print()
print(len(assertions))
variables.pop()
start_time = time.time()
x_related_assertions = find_assertions_related_to_var_names_optimized_dfs(solver, variables)
end_time = time.time()-start_time
print(end_time)
# print(x_related_assertions)
sum = 0
for k, v in x_related_assertions.items():
    sum += len(v)
    print(len(v))
print(sum)
print(len(assertions))
    # if len(v) == 0:
    #     print(k, v)
    # solver = Solver()
    # solver = Solver()
    # for a in v:
    #     solver.add(a)
    # r = solver.check()
    # print(r)
    # print(solver.model())
# x_related_assertions = find_assertions_related_to_var_names_optimized(solver, variables)
# print(x_related_assertions)
# solver = Solver()
# for a in x_related_assertions:
#     # solver.add(a)
#     print(a)
# r = solver.check()
# print(r)
# print(solver.model())

# opt = Optimize()
# # 寻找x的最小值
# opt.push()  # 保存当前约束状态
# opt.minimize(x)
# if opt.check() == sat:
#     print("Minimum value of x:", opt.model()[x])
# opt.pop()  # 恢复到之前的约束状态
#
# # 寻找x的最大值
# opt.push()  # 保存当前约束状态
# opt.maximize(x)
# if opt.check() == sat:
#     print("Maximum value of x:", opt.model()[x])
# opt.pop()  # 恢复到之前的约束状态
# smtlib_str_before, smtlib_str_after = split_at_check_sat(solver.to_smt2())
# new_constraint = "(assert (= {} (_ bv{} {})))\n".format('mem_1_263_8', 196, 8)
# smtlib_str = smtlib_str_before + new_constraint + smtlib_str_after
# assertions = parse_smt2_string(smtlib_str)
# solver = Solver()
# for a in assertions:
#     solver.add(a)
# r = solver.check()
# print(r)


# print(solver.model())
#     assertions_list.append(a)
#     solver.add(a)
# part = solver.assertions()
# print('*********************************')
# print(type(part))
# res = random.sample(assertions_list,int(len(assertions)*0.8))
# print(res)
# solver = Solver()
# for r in res:
#     solver.add(r)
# res = random.sample(smtlib_str)
# print(solver.to_smt2())
# print(type(assertions))
# # v_name = "v_name"
# # exec(f"{v_name} = Int('{variable_pred}')")
# # solver.add(v_name == selected_int)
# # print(solver)
# if solver.check() == sat:
#     # 如果存在满足约束的解，使用model()方法获取它
#     model = solver.model()
#     print(model)
# else:
#     print("没有找到满足所有约束的解")
#
