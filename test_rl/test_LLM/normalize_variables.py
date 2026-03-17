"""
变量归一化函数
类似于utils.py中的normalize_smt_str，对所有变量进行归一化替换
"""
import re


def count_variable_occurrences_in_content(content, variable):
    """
    统计单个变量在内容中出现的次数
    """
    pattern = r'\b' + re.escape(variable) + r'\b'
    matches = re.findall(pattern, content)
    return len(matches)


def normalize_all_variables(smtlib_str, variables):
    """
    对所有变量进行归一化替换，按照出现次数排序，替换为VAR_1, VAR_2, ..., VAR_N
    
    参数:
    - smtlib_str: 原始SMT文件内容
    - variables: 变量名列表
    
    返回:
    - normalized_content: 归一化后的内容
    - variable_mapping: {原始变量: 归一化变量} 的映射字典
    """
    # 统计每个变量的出现次数
    var_counts = {}
    for var in variables:
        var_counts[var] = count_variable_occurrences_in_content(smtlib_str, var)
    
    # 按照出现次数降序排序（出现次数多的排在前面）
    sorted_variables = sorted(variables, key=lambda v: var_counts.get(v, 0), reverse=True)
    
    # 生成映射字典 {原始变量: VAR_i}
    variable_mapping = {var: f"VAR_{i+1}" for i, var in enumerate(sorted_variables)}
    
    # 替换变量名（从长变量名开始替换，避免部分匹配的问题）
    # 按变量名长度降序排序，先替换长的变量名
    sorted_by_length = sorted(variable_mapping.items(), key=lambda x: len(x[0]), reverse=True)
    
    normalized_content = smtlib_str
    for original_var, normalized_var in sorted_by_length:
        pattern = r'\b' + re.escape(original_var) + r'\b'
        normalized_content = re.sub(pattern, normalized_var, normalized_content)
    
    return normalized_content, variable_mapping, var_counts

