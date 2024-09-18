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
                    # 获取最小化后的VAR1的值
                    print("Minimum value of VAR1:", opt.model()[var_nodes[var_name][-1]])
                else:
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
                    # 获取最小化后的VAR1的值
                    print("Maxmum value of VAR1:", opt.model()[var_nodes[var_name][-1]])
                else:
                    print("Optimization result is not satisfiable")


        # 重置results，以便下一次断言检查
        results = {var_name: False for var_name in var_names}
    return related_assertions_dict
