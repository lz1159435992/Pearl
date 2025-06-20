import json
import os
import time
from z3 import *
from test_script.utils import *
def solve(filepath):
    with open(file_path, 'r') as file:
        smtlib_str = file.read()
        dict_obj = json.loads(smtlib_str)
        smtlib_str = dict_obj['script']
    # print(smtlib_str)
    assertions = parse_smt2_string(smtlib_str)

    solver = Solver()
    for a in assertions:
        solver.add(a)
    # Visit each assertion to extract variables

    result, model, time_taken = solve_and_measure_time(solver, 99999999)
    if time_taken > 10:
        print(f"������� {result}, ����ʱ�䣺 {time_taken:.2f} ����ֵ�� {model},·����{filepath}")
#��Ҫ����·��
path = []
directory = '/home/nju/smt_collect/test'
path.append(directory)
#����·��
for directory in path:
  for dirpath, dirnames, filenames in os.walk(directory):
      for filename in filenames:
          file_path = os.path.join(dirpath, filename)
          solve(file_path)