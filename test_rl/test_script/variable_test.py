import re
from z3 import *
import json
variables = set()
def visit(expr):
    if is_const(expr) and expr.decl().kind() == Z3_OP_UNINTERPRETED:
        # Add only uninterpreted functions (which represent variables)
        variables.add(str(expr))
    else:
        # Recursively visit children for composite expressions
        for child in expr.children():
            visit(child)
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
            variables.append(match.group(1))

    return variables

if __name__ == '__main__':

    # file_path = '/home/lz/baidudisk/smt/buzybox_angr.tar.gz/single_test/readahead/readahead651389'
    # file_path = '/home/lz/baidudisk/smt/gnu_angr.tar.gz/single_test/who/who202348'
    for dirpath, dirnames, filenames in os.walk('/home/lz/baidudisk/smt/buzybox_angr.tar.gz/single_test'):
        for filename in filenames:
            # 构造完整的文件路径
            file_path = os.path.join(dirpath, filename)
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
            smtlib_str = dict_obj['script']
            assertions = parse_smt2_string(smtlib_str)

            variables = set()

            solver = Solver()
            for a in assertions:
                solver.add(a)

            # Extract variables from each assertion
            for a in assertions:
                visit(a)
            print(type(variables))
            # 调用函数并打印结果
            variables_2 = extract_variables_from_smt2_content(smtlib_str)
            variables_2 = set(variables_2)
            print(type(variables_2))
            if variables == variables_2:
                pass
            else:
                print('errorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr!!!!!!!!!')
