import json
import os
import re
import sqlite3
import time
from z3 import *

from pysmt.walkers.identitydag import IdentityDagWalker
from pysmt.smtlib.parser import SmtLibParser
from pysmt.operators import ALL_TYPES, AND, OR, LE, LT, BV_ULT, BV_ULE, BV_SLT, BV_SLE, BV_COMP, EQUALS, PLUS, MINUS, \
    TIMES, ITE, BV_NOT, BV_AND, BV_OR, BV_XOR, BV_NEG, BV_ADD, BV_SUB, BV_MUL, BV_UDIV, BV_UREM, DIV, POW
import argparse
import numpy as np
from pysmt.exceptions import PysmtTypeError
import os
# from test_rl.test_script.utils import parse_smt2_in_parts, process_smt_lib_string, fetch_data_as_dict, \
#     solve_and_measure_time, model_to_dict, load_dictionary, extract_variables_from_smt2_content, normalize_variables
from collections import defaultdict

from pysmt.environment import get_env, push_env, Environment

value_dict = {}
result_dict = {}
value_db_path = 'value_dictionary_nju.db'
value_table_name = 'value_dictionary_nju'
result_db_path = 'result_dictionary_nju.db'
result_table_name = 'result_dictionary_nju'


def load_dictionary(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


# 指定需要遍历的目录
def solve_and_measure_time(solver, timeout):
    solver.set("timeout", timeout)
    start_time = time.time()
    result = solver.check()
    stats = solver.statistics()
    # print(result)
    elapsed_time = stats.get_key_value('time')
    if result == sat:
        return "sat", solver.model(), elapsed_time
    elif result == unknown:
        return "unknown", None, elapsed_time
    else:
        return "unsat", None, elapsed_time


def model_to_dict(model):
    result = {}
    for var in model:
        result[str(var)] = str(model[var])
    return result


def solve(filepath, timeout):
    output_path = 'result_dict_no_increment_special_920.txt'
    if not os.path.exists(output_path):
        # 文件不存在时，创建文件
        result_dict = {}
        with open(output_path, 'w') as file:
            json.dump(result_dict, file, indent=4)
        print(f"文件 {output_path} 已创建。")
    else:
        result_dict = load_dictionary(output_path)
    if filepath not in result_dict.keys():
        # 键不存在，添加键值对
        with open(filepath, 'r') as file:
            # 璇诲彇鏂囦欢鎵€鏈夊唴瀹瑰埌涓€涓瓧绗︿覆
            smtlib_str = file.read()
        # try:
        #     # 灏咼SON瀛楃涓茶浆鎹负瀛楀吀
        #     dict_obj = json.loads(smtlib_str)
        #     # print("杞崲鍚庣殑瀛楀吀锛?, dict_obj)
        # except json.JSONDecodeError as e:
        #     print('failed', e)
        # #
        # smtlib_str = dict_obj['smt_script']
        # print(smtlib_str)
        assertions = parse_smt2_string(smtlib_str)
        solver = Solver()
        for a in assertions:
            solver.add(a)
        result, model, time_taken = solve_and_measure_time(solver, timeout)
        result_list = []
        # if result == sat:
        #     result = 'sat'
        # elif result == unknown:
        #     result = 'unknown'
        # else:
        #     result = 'unsat'
        result_list.append(result)
        result_list.append(time_taken)
        result_list.append(timeout)
        # result_dict[filepath] = result_list
        # print(type(model))
        if model:
            result_list.append(model_to_dict(model))
        else:
            result_list.append(None)
        result_dict[file_path] = result_list
        with open(output_path, 'w') as file:
            json.dump(result_dict, file, indent=4)


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

        self.special_operation_count = 0  # Store counts of logic operations
        self.all_operation_count = 0

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
        #判断特定的操作符
        self.all_operation_count += 1
        if formula.node_type() in [PLUS, MINUS, TIMES,  # LIA/LRA operators (13-15)
                                   LE, LT, EQUALS,  # LIA/LRA relations (16-18)
                                   ITE, BV_NOT, BV_AND, BV_OR, BV_XOR,  # Logical Operators on Bit (22-25)
                                   BV_ULT, BV_ULE,  # Unsigned Comparison (28-29)
                                   BV_NEG, BV_ADD, BV_SUB,  # Basic arithmetic (30-32)
                                   BV_MUL, BV_UDIV, BV_UREM,  # Division/Multiplication (33-35)
                                   BV_COMP, DIV,  # Arithmetic Division (62)
                                   POW]:  # Arithmetic Power (63)
            # Returns 1_1 if the arguments are                               #                       ]:
            self.special_operation_count += 1
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
                #收集指定的操作

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

    print(astBuilder.all_operation_count, astBuilder.special_operation_count)
    return astBuilder.special_operation_count/astBuilder.all_operation_count

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


if __name__ == '__main__':
    test_path = []
    # directory = '/home/yy/Downloads/smt/buzybox_angr.tar.gz/single_test'
    # test_path.append(directory)
    # directory = '/home/yy/Downloads/smt/gnu_angr.tar.gz/single_test'
    # test_path.append(directory)
    # directory = '/home/yy/Downloads/smt/gnu_KLEE/klee_bk/single_test'
    # test_path.append(directory)
    directory = '/home/lz/Downloads/non-incremental_Hierarchy/non-incremental'
    test_path.append(directory)
    # directory = '/home/lz/Downloads/incremental_Hierarchy/incremental'
    # test_path.append(directory)

    # 遍历目录
    for directory in test_path:
        path = []
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                # 构造完整的文件路径
                file_path = os.path.join(dirpath, filename)
                print(file_path)  # 或者进行其他操作
                if 'starexec_description.txt' in file_path or '52759_b3ecd2335fd16ec2eee2_9_UFDTBV' in file_path or 'sll-optional-1.i_1' in file_path:
                    print('NOTHING ')
                else:
                    with open(file_path, 'r') as file:
                        # 璇诲彇鏂囦欢鎵€鏈夊唴瀹瑰埌涓€涓瓧绗︿覆
                        smtlib_str = file.read()
                    # with open('/home/lz/PycharmProjects/Pearl/test_rl/ge_cons/auto_gen_v2.txt', 'r') as file:
                    #     result_dict = json.load(file)
                    # for k, v in result_dict.items():
                    #     smtlib_str = v[1]
                        p = normalize_smt_str(smtlib_str)
                        print(p)
                        if p > 0.65:
                            solve(file_path, 12000000)
                    # if '/QF_BV/' in file_path:
                    #     print('BV')
                    #     solve(file_path, 86400000)
