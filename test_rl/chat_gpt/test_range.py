import io
from collections import defaultdict
from pysmt.environment import Environment
from pysmt.smtlib.parser import SmtLibParser
from pysmt.walkers import IdentityDagWalker
from pysmt.operators import ALL_TYPES, AND, OR
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
        self.variable_logic_operation_count = None
        self.variable_clause_size = None
        self.constant_list = []  # 用于存储常量
        self.variable_bounds = defaultdict(lambda: {'lb': None, 'ub': None})  # 用于存储变量的上下界

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

            # 收集变量类型
            self.variable_type[formula] = formula.symbol_type()

            # 收集变量频率
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

            # 将常量添加到列表中
            if formula not in self.constant_list:
                self.constant_list.append(formula)  # 收集常量

        elif formula.node_id() not in self.id_to_counter:
            self.id_to_counter[formula.node_id()] = self.nodeCounter

            value = self.nodeCounter
            self.nodeCounter += 1
            self.add_node(formula)
        else:
            value = self.id_to_counter[formula.node_id()]

        return value

    def _push_with_children_to_stack(self, formula, **kwargs):
        """添加子公式到栈中。"""
        self.stack.append((True, formula))

        parenId = self.get_node_counter(formula, True)

        for s in self._get_children(formula):
            # 仅在未被记忆化时添加
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

                # 检查此子公式是否涉及常量
                for sub_s in self._get_children(formula):
                    if sub_s.is_constant():
                        if s not in self.variable_constant_clause_count:
                            self.variable_constant_clause_count[s] = set()
                        self.variable_constant_clause_count[s].add(parenId)
                        break

                # 收集逻辑操作次数
                self.variable_logic_operation_count[s] += 1  # 计数逻辑操作

                # 收集子公式大小
                self.variable_clause_size[s] = len(self._get_children(formula))

            key = self._get_key(s, **kwargs)
            if key not in self.memoization:
                self.stack.append((False, s))

    def visit(self, formula):
        if formula.is_gt() or formula.is_ge():
            left, right = formula.arg(0), formula.arg(1)
            if left.is_symbol() and right.is_constant():
                var = str(left)
                bound = right.constant_value()
                if formula.is_gt():
                    bound += 1  # x > 5 相当于 x >= 6
                if self.variable_bounds[var]['lb'] is None or bound > self.variable_bounds[var]['lb']:
                    self.variable_bounds[var]['lb'] = bound

        if formula.is_lt() or formula.is_le():
            left, right = formula.arg(0), formula.arg(1)
            if left.is_symbol() and right.is_constant():
                var = str(left)
                bound = right.constant_value()
                if formula.is_lt():
                    bound -= 1  # x < 5 相当于 x <= 4
                if self.variable_bounds[var]['ub'] is None or bound < self.variable_bounds[var]['ub']:
                    self.variable_bounds[var]['ub'] = bound

        # 递归访问子公式
        children = [self.visit(subf) for subf in formula.args()]
        for child in children:
            self.edges[0].append(len(self.nodes))
            self.edges[1].append(child)
            self.edge_attr.append(1)  # 假设我们对每个边使用相同的属性
        return len(self.nodes)


def normalize_smt_str_without_replace(smtlib_str):
    with Environment() as env:
        file_obj = io.StringIO(smtlib_str)
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

        # 输出变量属性
        variable_clause_count = {str(var): len(clauses) for var, clauses in astBuilder.variable_clause_count.items()}
        variable_constant_clause_count = {str(var): len(clauses) for var, clauses in
                                          astBuilder.variable_constant_clause_count.items()}
        variable_type = {str(var): str(var_type) for var, var_type in astBuilder.variable_type.items()}
        variable_frequency = {str(var): freq for var, freq in astBuilder.variable_frequency.items()}
        variable_logic_operation_count = {str(var): count for var, count in
                                          astBuilder.variable_logic_operation_count.items()}
        variable_clause_size = {str(var): size for var, size in astBuilder.variable_clause_size.items()}

        # 组合属性进行排序
        combined_attributes = {var: (
            variable_clause_size.get(var, 0),
            variable_clause_count.get(var, 0),
            variable_frequency.get(var, 0),
            variable_logic_operation_count.get(var, 0),  # 总逻辑操作次数
            variable_constant_clause_count.get(var, 0),
        ) for var in variable_clause_count.keys()}

        # 也可以直接提取变量名列表
        # 使用extract_variables_from_smt2_content(smtlib_str)

        # 基于组合属性对变量进行排序
        sorted_variables = sorted(combined_attributes.keys(), key=lambda x: (
            combined_attributes[x][0],  # 变量子公式大小
            combined_attributes[x][1],  # 变量到子公式数量
            combined_attributes[x][2],  # 变量频率
            combined_attributes[x][3],  # 总逻辑操作次数
            combined_attributes[x][4],  # 变量常量子公式数量
        ), reverse=True)

        # 提取常量并追加到结果中
        sorted_variables = [var for var in sorted_variables if variable_type[var] != 'Bool']

        # 输出变量的上下界信息
        variable_bounds = {var: bounds for var, bounds in astBuilder.variable_bounds.items()}

    return sorted_variables, variable_bounds


# 示例使用
smtlib_str = """
(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)
(assert (> x 5))
(assert (< y 10))
(assert (>= x 7))
(assert (<= y 8))
(check-sat)
"""

variables, bounds = normalize_smt_str_without_replace(smtlib_str)
print("变量排序:", variables)
print("变量上下界:", bounds)
