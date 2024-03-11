import json
from z3 import *
file_path = '/home/nju/Downloads/smt/gnu_angr.tar.gz/single_test/who/who53174'
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
assertions = parse_smt2_string(smtlib_str)

print(assertions)
for a in assertions:
    print(a)