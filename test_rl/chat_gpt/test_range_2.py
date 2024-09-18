from pysmt.fnode import FNode
import io
from collections import defaultdict
from pysmt.environment import Environment
from pysmt.smtlib.parser import SmtLibParser
from pysmt.walkers import IdentityDagWalker
from pysmt.operators import ALL_TYPES, AND, OR, BV_ULT, LT, LE, BV_ULE, BV_SLT, BV_SLE, BV_COMP, EQUALS
import numpy as np


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

                # Check for bounds
                # for sub_s in self._get_children(formula):
            key = self._get_key(s, **kwargs)
            if key not in self.memoization:
                self.stack.append((False, s))
        # 一个约束只判断一次
        print('*****************')
        print(formula.arg)
        print(len(formula.args()))
        print(formula.node_type())
        if len(formula.args()) == 2:
            left, right = formula.arg(0), formula.arg(1)
            # 获取变量间的大小关系
            print(left, left.node_type(), right, right.node_type(), formula.node_type())
            # if sub_s.is_constant():
            #     constant_value = sub_s.constant_value()
            if formula.node_type() in [LE, LT, BV_ULT, BV_ULE, BV_ULT, BV_SLT, BV_SLE]:
                # 左边变量小于常量
                if left.is_symbol() and right.is_constant():
                    constant_value = right.constant_value()
                    self.variable_bounds[left]['upper'].append(constant_value)
                # 右边变量小于常量
                elif right.is_symbol() and left.is_constant():
                    constant_value = left.constant_value()
                    self.variable_bounds[right]['lower'].append(constant_value)
                # 左边变量小于右边变量
                elif left.is_symbol() and right.is_symbol():
                    self.variable_bounds[left]['up_var'].append(right)
                # 考虑存在线性约束的情况  三个类型 PLUS, MINUS, TIMES  之后添加
            elif formula.node_type() in [BV_COMP, EQUALS]:
                if left.is_symbol() and right.is_constant():
                    constant_value = right.constant_value()
                    self.variable_bounds[left]['equal'].append(constant_value)
                elif right.is_symbol() and left.is_constant():
                    constant_value = left.constant_value()
                    self.variable_bounds[right]['equal'].append(constant_value)
                else:
                    self.variable_bounds[left]['equal_var'].append(right)


def normalize_smt_str_without_replace(smtlib_str):
    with Environment() as env:
        file_obj = io.StringIO(smtlib_str)
        # try:
        myParser = SmtLibParser(env)
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
        variable_bounds = {str(var): bounds for var, bounds in astBuilder.variable_bounds.items()}

        # Combine attributes for sorting
        combined_attributes = {var: (
            variable_clause_size.get(var, 0),
            variable_clause_count.get(var, 0),
            variable_frequency.get(var, 0),
            variable_logic_operation_count.get(var, 0),  # Total logic operations count
            variable_constant_clause_count.get(var, 0),
        ) for var in variable_clause_count.keys()}

        # Sort variables based on combined attributes
        sorted_variables = sorted(combined_attributes.keys(), key=lambda x: (
            combined_attributes[x][0],  # Variable Clause Sizes
            combined_attributes[x][1],  # Variable to Clause Count
            combined_attributes[x][2],  # Variable Frequencies
            combined_attributes[x][3],  # Total Logic Operations Count
            combined_attributes[x][4],  # Variable Constant Clause Count
        ), reverse=True)

        # Extract constants and append to the result
        sorted_variables = [var for var in sorted_variables if variable_type[var] != 'Bool']
        print(variable_bounds)
        variable_bounds = {var: variable_bounds[var] for var in sorted_variables}

    return sorted_variables, variable_bounds


# Constants and Operations
# 示例使用
smtlib_str = """
(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)
(assert (> x 5))
(assert (< x 10))
(assert (= x 9))
(assert (= x y))
(assert (< x y))
(assert (< y 10))
(assert (>= x 7))
(assert (<= y 8))
(assert (= x (+ y 2)))
(assert (< (+ x y) 20))
(check-sat)
"""

variables, bounds = normalize_smt_str_without_replace(smtlib_str)
print("变量排序:", variables)
print("变量上下界:", bounds)


def combine_intervals(constraints):
    # 初始化变量的最终区间
    final_intervals = {'x': None, 'y': None}

    # 处理变量x
    x_lower = max(constraints['x']['lower']) if constraints['x']['lower'] else None
    x_upper = min(constraints['x']['upper']) if constraints['x']['upper'] else None
    x_equal = constraints['x']['equal']
    x_low_var = [eval_expr(expr) for expr in constraints['x']['low_var']]
    x_up_var = [eval_expr(expr) for expr in constraints['x']['up_var']]
    x_equal_var = [eval_expr(expr) for expr in constraints['x']['equal_var']]

    # 确定x的最终区间
    if x_equal:
        final_intervals['x'] = (x_equal[0], x_equal[0])
    else:
        # 合并下界和上界
        x_interval = (x_lower, x_upper) if x_lower is not None and x_upper is not None else (
            x_lower,) if x_lower is not None else (x_upper,)
        # 考虑与其它变量相关的约束
        for var_val in x_low_var + x_up_var + x_equal_var:
            if var_val is not None:
                x_interval = (max(x_interval[0], var_val) if x_interval[0] is not None else var_val,
                              min(x_interval[1], var_val) if x_interval[1] is not None else var_val)
        final_intervals['x'] = x_interval

    # 对变量y重复相同的过程
    # ...（省略y的处理代码，与x的处理类似）

    return final_intervals


# 辅助函数，用于计算表达式
def eval_expr(expr):
    # 这里应该实现表达式的求值逻辑，例如使用eval()函数或相应的库
    # 注意：eval()不安全，实际使用时应避免或替换为安全的求值方法
    return eval(expr)


print(combine_intervals(bounds))
