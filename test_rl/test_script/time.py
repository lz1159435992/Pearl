import json
import random
import re
from z3 import *
import time


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
file_path = '/home/lz/baidudisk/smt/gnu_angr.tar.gz/single_test/seq/seq143541'
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
smtlib_str_before,smtlib_str_after = split_at_check_sat(smtlib_str)
print(smtlib_str_before,'************************************',smtlib_str_after)


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

type_info = find_var_declaration_in_string(smtlib_str,variable_pred)
print(type_info)
print(type(type_info))
if type_info == '_ BitVec 64':
    new_constraint = "(assert (= {} (_ bv{} 64)))\n".format(variable_pred,selected_int)
elif type_info == '_ BitVec 8':
    new_constraint = "(assert (= {} (_ bv{} 8)))\n".format(variable_pred,selected_int)
elif type_info == '_ BitVec 1008':
    new_constraint = "(assert (= {} (_ bv{} 1008)))\n".format(variable_pred,selected_int)
smtlib_str = smtlib_str_before + new_constraint + smtlib_str_after
assertions = parse_smt2_string(smtlib_str)
# assertions = process_smt_lib_string(smtlib_str)
variables = set()

solver = Solver()
assertions_list = []
for a in assertions:
    solver.add(a)
#     for i in a:
#         print()
x_related_assertions = find_assertions_related_to_var_name(solver, 'mem_1_263_8')
# r = solver.check()
# print(r)
# print(solver.model())

solver = Solver()
for a in x_related_assertions:
    solver.add(a)
    # print(a)
r = solver.check()
print(r)
print(solver.model())

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
smtlib_str_before, smtlib_str_after = split_at_check_sat(solver.to_smt2())
new_constraint = "(assert (= {} (_ bv{} {})))\n".format('mem_1_263_8', 196, 8)
smtlib_str = smtlib_str_before + new_constraint + smtlib_str_after
assertions = parse_smt2_string(smtlib_str)
solver = Solver()
for a in assertions:
    solver.add(a)
r = solver.check()
print(r)
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
