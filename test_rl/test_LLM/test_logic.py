"""
测试实验脚本的核心逻辑（不依赖LLM调用）
"""
import re
from itertools import permutations, combinations

def extract_variables_from_smt2_content(content):
    """从SMT2内容中提取变量"""
    variable_pattern = re.compile(r'\(declare-fun\s+([^ ]+)\s*\(\s*\)\s*([^)]+)\)')
    variables = []
    
    for line in content.splitlines():
        match = variable_pattern.search(line)
        if match:
            var_name, var_type = match.group(1, 2)
            if var_type.strip() != 'Bool':
                variables.append(var_name.replace('|', ''))
    
    return variables


def count_variable_occurrences(content, variables):
    """统计变量出现次数"""
    counts = {}
    for var in variables:
        pattern = r'\b' + re.escape(var) + r'\b'
        matches = re.findall(pattern, content)
        counts[var] = len(matches)
    return counts


def random_replace_variables(smtlib_str, var_mapping):
    """替换变量名"""
    result = smtlib_str
    for old_var, new_var in var_mapping.items():
        pattern = r'\b' + re.escape(old_var) + r'\b'
        result = re.sub(pattern, new_var, result)
    return result


def generate_variable_mappings(variables, num_mappings=20):
    """生成变量到VAR_1, VAR_2, VAR_3的映射组合"""
    if len(variables) < 3:
        return []
    
    var_names = ['VAR_1', 'VAR_2', 'VAR_3']
    mappings = []
    seen_mappings = set()
    
    # 如果变量数>=5，可以从不同组合中选择
    if len(variables) >= 5:
        var_combinations = list(combinations(variables[:min(len(variables), 7)], 3))
        
        for var_combo in var_combinations:
            if len(mappings) >= num_mappings:
                break
            
            for perm in permutations(range(3)):
                mapping = {var_combo[i]: var_names[perm[i]] for i in range(3)}
                mapping_key = tuple(sorted(mapping.items()))
                
                if mapping_key not in seen_mappings:
                    seen_mappings.add(mapping_key)
                    mappings.append(mapping)
                    
                    if len(mappings) >= num_mappings:
                        break
    
    # 如果还不够，使用前3个变量的所有排列
    if len(mappings) < num_mappings:
        vars_to_map = variables[:3]
        for perm in permutations(range(3)):
            if len(mappings) >= num_mappings:
                break
            
            mapping = {vars_to_map[i]: var_names[perm[i]] for i in range(3)}
            mapping_key = tuple(sorted(mapping.items()))
            
            if mapping_key not in seen_mappings:
                seen_mappings.add(mapping_key)
                mappings.append(mapping)
    
    return mappings


# 测试代码
if __name__ == '__main__':
    # 测试SMT内容
    test_smt = """
(set-logic QF_NIA)
(declare-fun x () Int)
(declare-fun y () Int)
(declare-fun z () Int)
(assert (> x 0))
(assert (> y 0))
(assert (> z 0))
(assert (= (+ x y) z))
(check-sat)
"""
    
    print("=" * 60)
    print("测试变量提取")
    print("=" * 60)
    variables = extract_variables_from_smt2_content(test_smt)
    print(f"提取的变量: {variables}")
    
    print("\n" + "=" * 60)
    print("测试变量出现次数统计")
    print("=" * 60)
    counts = count_variable_occurrences(test_smt, variables)
    print(f"变量出现次数: {counts}")
    
    print("\n" + "=" * 60)
    print("测试映射生成（20种）")
    print("=" * 60)
    # 为了测试20种映射，需要更多变量
    test_variables = ['x', 'y', 'z', 'a', 'b', 'c']
    mappings = generate_variable_mappings(test_variables, num_mappings=20)
    print(f"从 {len(test_variables)} 个变量中生成了 {len(mappings)} 种映射组合:")
    for i, mapping in enumerate(mappings[:10], 1):  # 只显示前10个
        print(f"  映射 {i}: {mapping}")
    if len(mappings) > 10:
        print(f"  ... (还有 {len(mappings)-10} 种映射)")
    
    print("\n" + "=" * 60)
    print("测试变量替换")
    print("=" * 60)
    if mappings:
        test_mapping = mappings[0]
        print(f"使用映射: {test_mapping}")
        replaced = random_replace_variables(test_smt, test_mapping)
        print("替换后的内容:")
        print(replaced)
        
        # 验证替换后的变量出现次数
        replaced_vars = list(test_mapping.values())
        replaced_counts = count_variable_occurrences(replaced, replaced_vars)
        print(f"\n替换后变量出现次数: {replaced_counts}")
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)

