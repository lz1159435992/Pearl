import json
import os
import time
from z3 import *
def solve_and_measure_time(solver, timeout):
    solver.set("timeout", timeout)
    start_time = time.time()
    result = solver.check()
    print(result)
    elapsed_time = time.time() - start_time
    if result == sat:
        return "sucess", solver.model(), elapsed_time
    elif result == unknown:
        return "unknown", None, elapsed_time
    else:
        return "failed", None, elapsed_time
def solve(filepath):
    with open(file_path, 'r') as file:
        smtlib_str = file.read()
        dict_obj = json.loads(smtlib_str)
        smtlib_str = dict_obj['script']
    # print(smtlib_str)
    assertions = parse_smt2_string(smtlib_str)
    def visit(expr):
        if is_const(expr) and expr.decl().kind() == Z3_OP_UNINTERPRETED:
            # Add only uninterpreted functions (which represent variables)
            variables.add(str(expr))
        else:
            # Recursively visit children for composite expressions
            for child in expr.children():
                visit(child)

    solver = Solver()
    for a in assertions:
        solver.add(a)
    # Visit each assertion to extract variables

    result, model, time_taken = solve_and_measure_time(solver, timeout)
    print(f"������� {result}, ����ʱ�䣺 {time_taken:.2f} ����ֵ�� {model},·����{filepath}")
#��Ҫ����·��
path = []
directory = '/home/lz/baidudisk/smt/buzybox_angr.tar.gz/single_test'
path.append(directory)
directory = '/home/lz/baidudisk/smt/gnu_angr.tar.gz/single_test'
path.append(directory)
directory = '/home/lz/baidudisk/smt/gnu_KLEE/klee_bk/single_test'
path.append(directory)
directory = '/home/lz/baidudisk/smt/smt-comp/QF_BV'
path.append(directory)
#����·��
for directory in path:
  for dirpath, dirnames, filenames in os.walk(directory):
      for filename in filenames:
          file_path = os.path.join(dirpath, filename)
          print(file_path) 
              with open(file_path, 'r') as file:
                  smtlib_str = file.read()
                  dict_obj = json.loads(smtlib_str)
              except json.JSONDecodeError as e:
                  print("解析错误�?, e)
              #
              # smtlib_str = dict_obj['script']
              # print(dict_obj)
              if 'time' in dict_obj.keys() and 'solving_time_dic' in dict_obj.keys():
                  if float(dict_obj['time']) > 300 or float(dict_obj['solving_time_dic']['z3'][0]) > 300:
                      path.append(file_path)
                      solve(file_path)
              elif 'time' in dict_obj.keys():
                  if float(dict_obj['time']) > 300:
                      path.append(file_path)
                      solve(file_path)
              elif 'solving_time_dic' in dict_obj.keys():
                  if float(dict_obj['solving_time_dic']) > 300:
                      path.append(file_path)
                      solve(file_path)
              # print(smtlib_str)
          except e:
              print('导入文件错误')
print('*********************************************************************************')
file_path = 'smtcomp.txt'
# 使用with语句打开文件以确保正确关�?
with open(file_path, 'w') as file:
    for s in path:
        # 写入字符串并添加换行�?
    file.write(s + '\n')





