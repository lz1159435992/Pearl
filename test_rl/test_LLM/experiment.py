"""
LLM变量选择实验脚本
验证LLM选择的变量是否与变量名映射组合或变量出现次数相关
"""
import os
import json
import random
import re
import sys
# 不再需要permutations，改为直接随机生成排列以避免内存问题
from collections import Counter
from pathlib import Path

# 添加项目路径
sys.path.append('/home/lz/PycharmProjects/Pearl')

from loguru import logger
from ollama import Client
from normalize_variables import normalize_all_variables

# 配置
TEST_JSON_PATH = '/home/lz/PycharmProjects/Pearl/test_rl/test_QF_NIA/cvc5_process_QF_NIA/QF_NIA_test.json'
LLM_HOST = 'http://172.29.7.221:32792'
LLM_MODEL = 'llama3.1:70b'
NUM_SAMPLES = 100
NUM_MAPPINGS = 50  # 每种映射组合的数量
MIN_VARIABLES = 5  # 最少变量数量要求
MIN_PERMUTATION_CHANGES_RATIO = 0.5  # 排列变化的最小比例（至少50%的变量位置发生变化）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 设置日志
logger.remove()
logger.add(os.path.join(SCRIPT_DIR, 'experiment.log'), rotation="10 MB", retention="7 days")
logger.add(lambda msg: print(msg, end=''), colorize=True)


def extract_variables_from_smt2_content(content):
    """
    从 SMT2 格式的字符串内容中提取变量名，排除布尔类型的变量。
    
    参数:
    - content: SMT2 格式的字符串内容。
    
    返回:
    - 非布尔类型变量名列表。
    """
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
    """
    统计每个变量在文件内容中出现的次数
    
    参数:
    - content: SMT文件内容
    - variables: 变量名列表
    
    返回:
    - dict: {变量名: 出现次数}
    """
    counts = {}
    for var in variables:
        # 使用单词边界匹配，避免匹配到包含变量名的其他字符串
        pattern = r'\b' + re.escape(var) + r'\b'
        matches = re.findall(pattern, content)
        counts[var] = len(matches)
    return counts


def random_replace_variables(smtlib_str, var_mapping):
    """
    根据给定的映射替换SMT文件中的变量名
    
    参数:
    - smtlib_str: 原始SMT文件内容
    - var_mapping: dict, {原始变量名: 新变量名}
    
    返回:
    - 替换后的SMT文件内容
    """
    result = smtlib_str
    for old_var, new_var in var_mapping.items():
        # 使用单词边界匹配
        pattern = r'\b' + re.escape(old_var) + r'\b'
        result = re.sub(pattern, new_var, result)
    return result


def count_permutation_changes(perm):
    """
    计算排列中变化的位置数（不是恒等映射的位置数）
    
    参数:
    - perm: 排列列表，例如[2, 0, 1]表示位置0->2, 1->0, 2->1
    
    返回:
    - int: 变化的位置数量
    """
    return sum(1 for i, val in enumerate(perm) if i != val)


def generate_full_variable_remappings(normalized_variables, num_mappings=50, min_changes_ratio=0.5):
    """
    对所有归一化变量生成全量重新映射（所有变量的排列）
    确保生成的排列尽可能多的变量发生变化（排除恒等排列，优先选择变化多的排列）
    
    参数:
    - normalized_variables: 归一化后的变量名列表（如['VAR_1', 'VAR_2', 'VAR_3', 'VAR_4', ...]）
    - num_mappings: 需要生成的映射数量（默认50）
    - min_changes_ratio: 最小变化比例（默认0.5，即至少50%的变量位置发生变化）
    
    返回:
    - list: 映射组合的列表，每个元素是 {归一化变量: 新的VAR_i} 的字典，包含所有变量
    """
    if len(normalized_variables) == 0:
        logger.warning("归一化变量列表为空")
        return []
    
    n = len(normalized_variables)
    target_names = [f'VAR_{i+1}' for i in range(n)]  # ['VAR_1', 'VAR_2', ..., 'VAR_N']
    min_changes = max(1, int(n * min_changes_ratio))  # 至少变化的位置数
    
    # 计算可能的排列数（不实际生成，避免内存问题）
    from math import factorial
    try:
        total_permutations = factorial(n)
    except OverflowError:
        total_permutations = float('inf')  # 对于非常大的n，阶乘会溢出
    
    logger.info(f"共有{total_permutations}种可能的全量映射（{n}!），将随机生成{num_mappings}种（至少{min_changes}/{n}个变量位置发生变化）")
    
    # 对于大量变量，直接随机生成排列，而不是先生成所有排列
    import random
    selected_perms = []
    seen_perms = set()
    
    # 随机生成num_mappings个不同的排列
    max_attempts = num_mappings * 2000   # 增加尝试次数，因为现在有额外的约束
    attempts = 0
    
    while len(selected_perms) < num_mappings and attempts < max_attempts:
        attempts += 1
        # 生成一个随机排列
        perm = list(range(n))
        random.shuffle(perm)
        
        # 检查变化数量是否满足要求
        changes = count_permutation_changes(perm)
        if changes < min_changes:
            continue  # 变化太少，跳过这个排列
        
        perm_tuple = tuple(perm)
        
        if perm_tuple not in seen_perms:
            seen_perms.add(perm_tuple)
            selected_perms.append(perm_tuple)
    
    if len(selected_perms) < num_mappings:
        logger.warning(f"只生成了{len(selected_perms)}种不同的排列（尝试了{attempts}次，要求至少{min_changes}个变化）")
    
    # 为每个选中的排列生成映射
    mappings = []
    for perm in selected_perms:
        # 创建完整映射：每个归一化变量映射到新的VAR_i
        mapping = {normalized_variables[i]: target_names[perm[i]] for i in range(n)}
        mappings.append(mapping)
    
    logger.info(f"生成了{len(mappings)}种全量重新映射")
    return mappings


def create_prompt_for_variable_selection(smt_content, variables):
    """
    创建用于选择最应该首先具体化变量名的prompt
    
    参数:
    - smt_content: SMT文件内容
    - variables: 变量名列表（标准化后的VAR_1, VAR_2, ..., VAR_N）
    
    返回:
    - system_message和user_message的字典
    """
    var_str = ', '.join(variables)
    
    system_message = {
        "role": "system",
        "content": """I have an SMT file written in SMT-LIB format, containing variable declarations, logical expressions, 
and constraints. Please help me analyze this file to identify which variable should be concretized first 
to simplify the constraints and make them easier to solve.

Here's what I need:
1. Analyze the SMT content, using logical reasoning and heuristic methods to determine which variable, 
   if concretized first, would most effectively simplify the constraints.
2. Consider factors such as:
   - The complexity of expressions involving each variable
   - The number of constraints each variable appears in
   - The frequency of each variable's occurrence in the file
   - The structural impact of concretizing each variable on the overall constraint system
3. Select ONE variable from the provided variable list that should be concretized first.

Variables need to be selected from the following variables:
<""" + var_str + """>

Your output should be ONLY the variable name (e.g., VAR_1, VAR_2, or VAR_3), without any explanation, 
JSON format, or additional text. Just output the variable name."""
    }
    
    user_message = {
        "role": "user",
        "content": f"""Here is the SMT file content:

{smt_content}

Analyze the constraints and determine which variable should be concretized first to simplify the problem. 
The available variables are: {var_str}

Output ONLY the variable name (one of: {var_str})."""
    }
    
    return {
        "system": system_message,
        "user": user_message
    }


def create_prompt_for_second_variable_selection(smt_content, variables, first_selected_var):
    """
    创建用于选择第二个应该具体化变量名的prompt（在第一个变量已经被选中的基础上）
    
    参数:
    - smt_content: SMT文件内容
    - variables: 变量名列表（标准化后的VAR_1, VAR_2, ..., VAR_N）
    - first_selected_var: 第一个被选中的变量名
    
    返回:
    - system_message和user_message的字典
    """
    # 排除第一个已选中的变量
    remaining_variables = [v for v in variables if v != first_selected_var]
    var_str = ', '.join(remaining_variables)
    
    system_message = {
        "role": "system",
        "content": """I have an SMT file written in SMT-LIB format, containing variable declarations, logical expressions, 
and constraints. I have already selected one variable ({first_var}) to be concretized first. 
Now I need to identify which variable should be concretized next (second) to further simplify the constraints.

Here's what I need:
1. Analyze the SMT content, considering that {first_var} has already been selected for concretization.
2. Using logical reasoning and heuristic methods, determine which variable from the remaining variables 
   should be concretized next (second) to most effectively simplify the constraints.
3. Consider factors such as:
   - The complexity of expressions involving each remaining variable
   - The number of constraints each remaining variable appears in
   - The frequency of each remaining variable's occurrence in the file
   - How concretizing each remaining variable would interact with the first selected variable ({first_var})
   - The structural impact of concretizing each remaining variable on the overall constraint system
4. Select ONE variable from the remaining variable list that should be concretized second.

The remaining variables that can be selected from are:
<""" + var_str + """>

Your output should be ONLY the variable name (e.g., VAR_1, VAR_2, or VAR_3), without any explanation, 
JSON format, or additional text. Just output the variable name.""".format(first_var=first_selected_var)
    }
    
    user_message = {
        "role": "user",
        "content": f"""Here is the SMT file content:

{smt_content}

The variable {first_selected_var} has already been selected to be concretized first. 
Now analyze the constraints and determine which variable should be concretized second to further simplify the problem.
The remaining available variables are: {var_str}

Output ONLY the variable name (one of: {var_str})."""
    }
    
    return {
        "system": system_message,
        "user": user_message
    }


def query_llm_for_variable_selection(smt_content, variables, llm_host, llm_model, first_selected_var=None):
    """
    使用LLM选择最应该具体化的变量名
    
    参数:
    - smt_content: SMT文件内容
    - variables: 变量名列表
    - llm_host: LLM服务器地址
    - llm_model: LLM模型名称
    - first_selected_var: 如果提供，则用于第二次选择（第一个已选中的变量）
    
    返回:
    - str: LLM选择的变量名，如果出错返回None
    """
    try:
        if first_selected_var is None:
            # 第一次选择
            prompt_dict = create_prompt_for_variable_selection(smt_content, variables)
            valid_variables = variables
        else:
            # 第二次选择
            prompt_dict = create_prompt_for_second_variable_selection(smt_content, variables, first_selected_var)
            valid_variables = [v for v in variables if v != first_selected_var]
        
        client = Client(host=llm_host)
        response = client.chat(
            model=llm_model,
            messages=[prompt_dict["system"], prompt_dict["user"]],
            options={"temperature": 0.7},
            stream=True,
        )
        
        # 处理流式响应
        responses = []
        for chunk in response:
            if 'message' in chunk and 'content' in chunk['message']:
                responses.append(chunk['message']['content'])
        
        # 提取变量名
        result = ''.join(responses).strip()
        
        # 清理结果，只保留变量名
        result = re.sub(r'[^\w_]', '', result)  # 移除非字母数字下划线字符
        
        # 验证结果是否是有效的变量名
        if result in valid_variables:
            return result
        else:
            logger.warning(f"LLM返回了无效变量名: {result}, 有效变量: {valid_variables}")
            # 尝试从结果中提取变量名
            for var in valid_variables:
                if var in result or result in var:
                    return var
            return None
            
    except Exception as e:
        logger.error(f"LLM查询出错: {e}")
        return None


def run_experiment():
    """
    运行主实验
    """
    logger.info("="*80)
    logger.info("开始LLM变量选择实验")
    logger.info("="*80)
    
    # 1. 加载测试集
    logger.info(f"加载测试集: {TEST_JSON_PATH}")
    with open(TEST_JSON_PATH, 'r') as f:
        test_data = json.load(f)
    
    # 2. 筛选变量数量大于5的文件，然后随机选择
    logger.info("筛选变量数量大于5的文件...")
    files_with_sufficient_vars = []
    
    for file_path in test_data.keys():
        try:
            if not os.path.exists(file_path):
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                smt_content = f.read()
            
            variables = extract_variables_from_smt2_content(smt_content)
            
            if len(variables) > MIN_VARIABLES:
                files_with_sufficient_vars.append(file_path)
        except Exception as e:
            logger.debug(f"检查文件时出错 {file_path}: {e}")
            continue
    
    logger.info(f"找到{len(files_with_sufficient_vars)}个变量数量>{MIN_VARIABLES}的文件")
    
    # 随机选择
    if len(files_with_sufficient_vars) < NUM_SAMPLES:
        logger.warning(f"符合条件的文件只有{len(files_with_sufficient_vars)}个，少于请求的{NUM_SAMPLES}个")
        selected_files = files_with_sufficient_vars
    else:
        selected_files = random.sample(files_with_sufficient_vars, NUM_SAMPLES)
    
    logger.info(f"随机选择了{len(selected_files)}个文件")
    
    # 3. 创建输出目录
    output_dir = os.path.join(SCRIPT_DIR, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    replaced_files_dir = os.path.join(output_dir, 'replaced_files')
    os.makedirs(replaced_files_dir, exist_ok=True)
    
    # 4. 存储实验结果
    results = []
    
    # 5. 处理每个文件
    for idx, file_path in enumerate(selected_files, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"处理文件 {idx}/{len(selected_files)}: {os.path.basename(file_path)}")
        logger.info(f"{'='*80}")
        
        try:
            # 读取SMT文件
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在，跳过: {file_path}")
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                smt_content = f.read()
            
            # 提取变量
            original_variables = extract_variables_from_smt2_content(smt_content)
            
            if len(original_variables) <= MIN_VARIABLES:
                logger.warning(f"变量数量不足 ({len(original_variables)}个，需要>{MIN_VARIABLES}个)，跳过此文件")
                continue
            
            logger.info(f"原始变量数量: {len(original_variables)}")
            logger.info(f"原始变量: {original_variables[:5]}... (显示前5个)")
            
            # 第一步：对所有变量进行归一化替换
            logger.info("步骤1: 对所有变量进行归一化替换...")
            normalized_content, normalization_mapping, original_var_counts = normalize_all_variables(
                smt_content, original_variables
            )
            
            # 获取归一化后的变量列表（按照重要性排序：VAR_1, VAR_2, ..., VAR_N）
            normalized_variables = list(normalization_mapping.values())
            logger.info(f"归一化后的变量: {normalized_variables[:5]}... (共{len(normalized_variables)}个)")
            logger.info(f"归一化映射示例: {dict(list(normalization_mapping.items())[:3])}")
            
            # 第二步：对所有归一化变量生成全量重新映射（50种不同的排列，尽可能多的变量发生变化）
            logger.info(f"步骤2: 对所有{len(normalized_variables)}个归一化变量生成全量重新映射...")
            remappings = generate_full_variable_remappings(normalized_variables, NUM_MAPPINGS, MIN_PERMUTATION_CHANGES_RATIO)
            logger.info(f"生成了{len(remappings)}种全量重新映射")
            
            # 处理每种重新映射组合
            for mapping_idx, remapping in enumerate(remappings, 1):
                logger.info(f"\n--- 重新映射组合 {mapping_idx}/{len(remappings)} ---")
                logger.info(f"重新映射: {remapping}")
                
                # 第三步：应用重新映射到归一化后的内容
                # remapping是完整的映射：{所有归一化变量: 新的VAR_i}
                # 例如（5个变量）: {'VAR_1': 'VAR_2', 'VAR_2': 'VAR_1', 'VAR_3': 'VAR_3', 'VAR_4': 'VAR_5', 'VAR_5': 'VAR_4'}
                # 使用两步替换避免变量名冲突：
                # 1. 先替换为临时变量名
                # 2. 再将临时变量名替换为目标变量名
                
                # 生成临时变量名（使用VAR_TEMP_i格式）
                temp_var_map = {}
                for idx, (normalized_var, new_var) in enumerate(remapping.items()):
                    temp_var = f'VAR_TEMP_{idx}'
                    temp_var_map[normalized_var] = (temp_var, new_var)
                
                final_content = normalized_content
                
                # 第一步：替换为临时变量名
                for normalized_var, (temp_var, _) in temp_var_map.items():
                    pattern = r'\b' + re.escape(normalized_var) + r'\b'
                    final_content = re.sub(pattern, temp_var, final_content)
                
                # 第二步：替换为目标变量名
                for normalized_var, (temp_var, new_var) in temp_var_map.items():
                    pattern = r'\b' + re.escape(temp_var) + r'\b'
                    final_content = re.sub(pattern, new_var, final_content)
                
                # 获取最终文件中的所有变量（所有变量都被重新映射了）
                # final_content中应该包含VAR_1, VAR_2, ..., VAR_N（所有变量）
                final_variables_in_file = [f'VAR_{i+1}' for i in range(len(normalized_variables))]
                
                # 构建反向归一化映射（归一化变量 -> 原始变量），用于后续查找
                reverse_normalization = {v: k for k, v in normalization_mapping.items()}
                
                # 找出映射到这些最终变量的归一化变量及其对应的原始变量
                # 按照final_variables_in_file的顺序（VAR_1, VAR_2, ..., VAR_N）进行查找
                mapped_normalized_vars = []  # 所有归一化变量（按最终变量顺序）
                mapped_original_vars = []    # 所有原始变量（按最终变量顺序）
                for final_var in final_variables_in_file:
                    # 找到映射到这个final_var的归一化变量
                    for norm_var, mapped_var in remapping.items():
                        if mapped_var == final_var:
                            mapped_normalized_vars.append(norm_var)
                            mapped_original_vars.append(reverse_normalization.get(norm_var))
                            break
                
                var_counts = {var: original_var_counts.get(var, 0) for var in mapped_original_vars if var}
                
                # 生成新文件名
                file_basename = os.path.splitext(os.path.basename(file_path))[0]
                new_filename = f"{file_basename}_remapping_{mapping_idx}.smt2"
                new_filepath = os.path.join(replaced_files_dir, new_filename)
                
                # 保存最终替换后的文件
                with open(new_filepath, 'w', encoding='utf-8') as f:
                    f.write(final_content)
                
                # 提取最终文件中的所有变量（所有变量都被重新映射，变量名为VAR_1, VAR_2, ..., VAR_N）
                final_variables = final_variables_in_file  # ['VAR_1', 'VAR_2', ..., 'VAR_N']
                
                # 统计最终变量的出现次数（统计所有变量）
                final_var_counts = count_variable_occurrences(final_content, final_variables)
                logger.info(f"最终变量出现次数: {final_var_counts}")
                
                # 第一次调用LLM选择变量
                logger.info("第一次调用LLM选择变量...")
                first_selected_var = query_llm_for_variable_selection(
                    final_content, 
                    final_variables, 
                    LLM_HOST, 
                    LLM_MODEL,
                    first_selected_var=None  # 第一次选择
                )
                
                if first_selected_var:
                    logger.info(f"LLM第一次选择的变量: {first_selected_var}")
                else:
                    logger.warning("LLM第一次选择未能返回有效变量名")
                
                # 第二次调用LLM选择变量（在第一个变量已选中的基础上）
                second_selected_var = None
                if first_selected_var:
                    logger.info("第二次调用LLM选择变量...")
                    second_selected_var = query_llm_for_variable_selection(
                        final_content,
                        final_variables,
                        LLM_HOST,
                        LLM_MODEL,
                        first_selected_var=first_selected_var  # 第二次选择
                    )
                    
                    if second_selected_var:
                        logger.info(f"LLM第二次选择的变量: {second_selected_var}")
                    else:
                        logger.warning("LLM第二次选择未能返回有效变量名")
                
                # 找出第一次选择的VAR_i对应的原始变量
                # 路径：LLM选择的VAR_i -> 重新映射中的归一化变量 -> 原始变量
                first_selected_original_var = None
                first_selected_normalized_var = None
                if first_selected_var:
                    # 第一步：找到重新映射中对应的归一化变量（哪个归一化变量映射到了选中的VAR_i）
                    for norm_var, mapped_var in remapping.items():
                        if mapped_var == first_selected_var:
                            first_selected_normalized_var = norm_var
                            break
                    
                    # 第二步：从归一化映射找到原始变量（使用之前构建的reverse_normalization）
                    if first_selected_normalized_var:
                        first_selected_original_var = reverse_normalization.get(first_selected_normalized_var)
                
                # 找出第二次选择的VAR_i对应的原始变量
                second_selected_original_var = None
                second_selected_normalized_var = None
                if second_selected_var:
                    # 第一步：找到重新映射中对应的归一化变量
                    for norm_var, mapped_var in remapping.items():
                        if mapped_var == second_selected_var:
                            second_selected_normalized_var = norm_var
                            break
                    
                    # 第二步：从归一化映射找到原始变量
                    if second_selected_normalized_var:
                        second_selected_original_var = reverse_normalization.get(second_selected_normalized_var)
                
                # 记录结果
                result_entry = {
                    "original_file": file_path,
                    "file_index": idx,
                    "remapping_index": mapping_idx,
                    "normalization_mapping": normalization_mapping,  # {原始变量: 归一化变量}
                    "remapping": remapping,  # {归一化变量: 新的VAR_i}
                    "original_variables": mapped_original_vars,  # 当前重新映射使用的原始变量
                    "all_original_variables": original_variables,  # 所有原始变量
                    "normalized_variables": mapped_normalized_vars,  # 当前重新映射使用的归一化变量
                    "all_normalized_variables": normalized_variables,  # 所有归一化变量
                    "final_variables": final_variables,  # 最终变量（VAR_1, VAR_2, ..., VAR_N，所有变量）
                    "original_var_counts": var_counts,  # 当前映射变量的原始出现次数
                    "all_original_var_counts": original_var_counts,  # 所有变量的原始出现次数
                    "final_var_counts": final_var_counts,  # 最终变量的出现次数
                    # 第一次选择
                    "first_llm_selected_var": first_selected_var,  # LLM第一次选择的VAR_i
                    "first_selected_normalized_var": first_selected_normalized_var,  # 对应的归一化变量
                    "first_selected_original_var": first_selected_original_var,  # 对应的原始变量
                    "first_selected_original_var_count": original_var_counts.get(first_selected_original_var) if first_selected_original_var else None,
                    # 第二次选择
                    "second_llm_selected_var": second_selected_var,  # LLM第二次选择的VAR_i
                    "second_selected_normalized_var": second_selected_normalized_var,  # 对应的归一化变量
                    "second_selected_original_var": second_selected_original_var,  # 对应的原始变量
                    "second_selected_original_var_count": original_var_counts.get(second_selected_original_var) if second_selected_original_var else None,
                    # 兼容性字段（保留旧的字段名，用于向后兼容）
                    "llm_selected_var": first_selected_var,  # 向后兼容
                    "selected_normalized_var": first_selected_normalized_var,  # 向后兼容
                    "selected_original_var": first_selected_original_var,  # 向后兼容
                    "selected_original_var_count": original_var_counts.get(first_selected_original_var) if first_selected_original_var else None,  # 向后兼容
                    "replaced_file": new_filepath,
                }
                results.append(result_entry)
                
        except Exception as e:
            logger.error(f"处理文件时出错: {file_path}, 错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # 6. 保存结果
    results_file = os.path.join(output_dir, 'experiment_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"实验完成！结果已保存到: {results_file}")
    logger.info(f"{'='*80}\n")
    
    # 7. 生成分析报告
    generate_analysis_report(results, output_dir)
    
    # 8. 生成可视化图表
    generate_visualization_plots(results, output_dir)
    
    return results


def generate_analysis_report(results, output_dir):
    """
    生成分析报告
    """
    logger.info("\n生成分析报告...")
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("LLM变量选择实验分析报告")
    report_lines.append("="*80)
    report_lines.append("")
    
    # 统计信息
    total_cases = len(results)
    successful_cases = sum(1 for r in results if r['llm_selected_var'] is not None)
    report_lines.append(f"总测试用例数: {total_cases}")
    report_lines.append(f"成功获取LLM选择: {successful_cases}")
    report_lines.append("")
    
    # 按文件分组分析
    files_dict = {}
    for result in results:
        file_idx = result['file_index']
        if file_idx not in files_dict:
            files_dict[file_idx] = []
        files_dict[file_idx].append(result)
    
    report_lines.append("="*80)
    report_lines.append("按文件分析")
    report_lines.append("="*80)
    
    for file_idx, file_results in sorted(files_dict.items()):
        report_lines.append(f"\n文件 {file_idx}:")
        original_file = file_results[0]['original_file']
        report_lines.append(f"  路径: {os.path.basename(original_file)}")
        report_lines.append(f"  所有原始变量: {file_results[0].get('all_original_variables', [])}")
        report_lines.append(f"  所有归一化变量: {file_results[0].get('all_normalized_variables', [])}")
        report_lines.append(f"  重新映射组合数: {len(file_results)}")
        report_lines.append("")
        
        # 分析每种重新映射组合的选择
        for result in file_results:
            remapping_idx = result.get('remapping_index', result.get('mapping_index', 'N/A'))
            remapping = result.get('remapping', result.get('mapping', {}))
            normalization_mapping = result.get('normalization_mapping', {})
            selected_var = result['llm_selected_var']
            final_counts = result.get('final_var_counts', {})
            
            report_lines.append(f"  重新映射组合 {remapping_idx}/{len(file_results)}:")
            report_lines.append(f"    使用的归一化变量: {result.get('normalized_variables', 'N/A')}")
            report_lines.append(f"    使用的原始变量: {result.get('original_variables', 'N/A')}")
            report_lines.append(f"    重新映射关系: {remapping}")
            report_lines.append(f"    原始变量出现次数: {result.get('original_var_counts', {})}")
            report_lines.append(f"    最终变量出现次数: {final_counts}")
            report_lines.append(f"    LLM选择: {selected_var}")
            
            # 找出选择的变量对应的原始变量
            if result.get('selected_original_var'):
                orig_var = result['selected_original_var']
                norm_var = result.get('selected_normalized_var', 'N/A')
                orig_count = result.get('selected_original_var_count', 'N/A')
                report_lines.append(f"    对应归一化变量: {norm_var}")
                report_lines.append(f"    对应原始变量: {orig_var} (出现{orig_count}次)")
            report_lines.append("")
    
    # 分析变量出现次数与选择的关系
    report_lines.append("="*80)
    report_lines.append("变量出现次数与LLM选择的关系分析")
    report_lines.append("="*80)
    
    # 统计：选择最多出现次数的变量的情况
    count_correct = 0  # 选择出现次数最多的变量
    total_with_counts = 0
    
    for result in results:
        final_counts = result.get('final_var_counts', {})
        if result.get('llm_selected_var') and final_counts:
            max_count_var = max(final_counts.items(), key=lambda x: x[1])[0]
            
            if result['llm_selected_var'] == max_count_var:
                count_correct += 1
            total_with_counts += 1
    
    if total_with_counts > 0:
        accuracy = count_correct / total_with_counts * 100
        report_lines.append(f"LLM选择出现次数最多变量的准确率: {accuracy:.2f}% ({count_correct}/{total_with_counts})")
    
    # 分析映射组合对选择的影响
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("映射组合对选择的影响")
    report_lines.append("="*80)
    
    # 统计每个VAR_i被选择的次数
    var_selection_counts = {'VAR_1': 0, 'VAR_2': 0, 'VAR_3': 0}
    for result in results:
        selected_var = result.get('llm_selected_var')
        if selected_var:
            if selected_var in var_selection_counts:
                var_selection_counts[selected_var] += 1
    
    report_lines.append("各变量（VAR_i）被选择的次数:")
    for var, count in var_selection_counts.items():
        percentage = count / successful_cases * 100 if successful_cases > 0 else 0
        report_lines.append(f"  {var}: {count}次 ({percentage:.1f}%)")
    
    # 分析：对于同一个文件的不同重新映射组合，是否选择了一致的原始变量
    report_lines.append("")
    report_lines.append("同一文件不同重新映射组合的一致性分析:")
    consistency_analysis = {}
    
    for file_idx, file_results in sorted(files_dict.items()):
        # 对于每个文件，收集所有重新映射组合选择的原始变量
        selected_original_vars = []
        for result in file_results:
            if result.get('selected_original_var'):
                selected_original_vars.append(result['selected_original_var'])
        
        if len(selected_original_vars) > 0:
            # 计算一致性：是否所有重新映射组合都选择了相同的原始变量
            unique_selections = set(selected_original_vars)
            is_consistent = len(unique_selections) == 1
            
            consistency_analysis[file_idx] = {
                'total_remappings': len(file_results),
                'selected_count': len(selected_original_vars),
                'unique_selections': list(unique_selections),
                'is_consistent': is_consistent
            }
            
            report_lines.append(f"  文件 {file_idx}:")
            report_lines.append(f"    选择的原始变量: {selected_original_vars}")
            report_lines.append(f"    唯一选择: {unique_selections}")
            report_lines.append(f"    一致性: {'是' if is_consistent else '否'}")
    
    # 统计一致性
    consistent_files = sum(1 for v in consistency_analysis.values() if v['is_consistent'])
    report_lines.append("")
    report_lines.append(f"一致性统计: {consistent_files}/{len(consistency_analysis)} 个文件在所有重新映射组合中选择了一致的原始变量")
    
    # 保存报告
    report_file = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"分析报告已保存到: {report_file}")
    
    # 打印摘要
    print("\n" + "\n".join(report_lines[:50]))  # 打印前50行


def generate_visualization_plots(results, output_dir):
    """
    生成可视化图表：一次选择和二次选择中选中的变量和其在原文中出现次数的关系
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
        return
    
    logger.info("\n生成可视化图表...")
    
    # 收集数据
    first_selected_counts = []  # 第一次选择的变量在原文中的出现次数
    second_selected_counts = []  # 第二次选择的变量在原文中的出现次数
    
    for result in results:
        # 第一次选择
        first_count = result.get('first_selected_original_var_count')
        if first_count is not None:
            first_selected_counts.append(first_count)
        
        # 第二次选择
        second_count = result.get('second_selected_original_var_count')
        if second_count is not None:
            second_selected_counts.append(second_count)
    
    if not first_selected_percentages and not second_selected_percentages:
        logger.warning("没有足够的数据生成图表")
        return
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LLM变量选择与变量出现次数百分比的关系分析', fontsize=16, fontweight='bold')
    
    # 1. 第一次选择：变量出现次数百分比分布直方图
    ax1 = axes[0, 0]
    if first_selected_percentages:
        ax1.hist(first_selected_percentages, bins=30, alpha=0.7, color='#1f77b4', edgecolor='black')
        ax1.set_xlabel('变量出现次数百分比 (%)', fontsize=11)
        ax1.set_ylabel('选择频次', fontsize=11)
        ax1.set_title('第一次选择：变量出现次数百分比分布', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        mean_first = np.mean(first_selected_percentages)
        ax1.axvline(mean_first, color='red', linestyle='--', 
                   label=f'平均值: {mean_first:.2f}%')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('第一次选择：变量出现次数百分比分布', fontsize=12)
    
    # 2. 第二次选择：变量出现次数百分比分布直方图
    ax2 = axes[0, 1]
    if second_selected_percentages:
        ax2.hist(second_selected_percentages, bins=30, alpha=0.7, color='#ff7f0e', edgecolor='black')
        ax2.set_xlabel('变量出现次数百分比 (%)', fontsize=11)
        ax2.set_ylabel('选择频次', fontsize=11)
        ax2.set_title('第二次选择：变量出现次数百分比分布', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        mean_second = np.mean(second_selected_percentages)
        ax2.axvline(mean_second, color='red', linestyle='--', 
                   label=f'平均值: {mean_second:.2f}%')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('第二次选择：变量出现次数百分比分布', fontsize=12)
    
    # 3. 第一次 vs 第二次选择的箱线图比较
    ax3 = axes[1, 0]
    if first_selected_percentages and second_selected_percentages:
        box_data = [first_selected_percentages, second_selected_percentages]
        bp = ax3.boxplot(box_data, labels=['第一次选择', '第二次选择'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#1f77b4')
        bp['boxes'][1].set_facecolor('#ff7f0e')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_alpha(0.7)
        ax3.set_ylabel('变量出现次数百分比 (%)', fontsize=11)
        ax3.set_title('第一次 vs 第二次选择：出现次数百分比比较（箱线图）', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('第一次 vs 第二次选择：出现次数百分比比较', fontsize=12)
    
    # 4. 散点图：第一次选择 vs 第二次选择（如果有配对数据）
    ax4 = axes[1, 1]
    paired_first = []
    paired_second = []
    for result in results:
        all_counts_dict = result.get('all_original_var_counts', {})
        if not all_counts_dict:
            continue
        total_counts = sum(all_counts_dict.values())
        if total_counts == 0:
            continue
        
        first_count = result.get('first_selected_original_var_count')
        second_count = result.get('second_selected_original_var_count')
        if first_count is not None and second_count is not None:
            first_percentage = (first_count / total_counts) * 100
            second_percentage = (second_count / total_counts) * 100
            paired_first.append(first_percentage)
            paired_second.append(second_percentage)
    
    if paired_first and paired_second:
        ax4.scatter(paired_first, paired_second, alpha=0.5, s=50, color='green')
        ax4.set_xlabel('第一次选择变量的出现次数百分比 (%)', fontsize=11)
        ax4.set_ylabel('第二次选择变量的出现次数百分比 (%)', fontsize=11)
        ax4.set_title('第一次 vs 第二次选择：出现次数百分比相关性', fontsize=12, fontweight='bold')
        
        # 添加对角线（y=x）
        max_val = max(max(paired_first), max(paired_second))
        ax4.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
        
        # 计算相关系数
        if len(paired_first) > 1:
            corr = np.corrcoef(paired_first, paired_second)[0, 1]
            ax4.text(0.05, 0.95, f'相关系数: {corr:.3f}', transform=ax4.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, '配对数据不足', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('第一次 vs 第二次选择：出现次数百分比相关性', fontsize=12)
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = os.path.join(output_dir, 'variable_selection_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"可视化图表已保存到: {plot_file}")
    
    plt.close()
    
    # 生成详细统计摘要
    stats_lines = []
    stats_lines.append("="*80)
    stats_lines.append("变量选择统计分析摘要")
    stats_lines.append("="*80)
    stats_lines.append("")
    
    # 计算变量排名（1=最高频）
    first_ranks = []
    second_ranks = []
    
    for result in results:
        all_counts_dict = result.get('all_original_var_counts', {})
        if not all_counts_dict:
            continue
        
        all_counts = sorted(all_counts_dict.values(), reverse=True)
        if len(all_counts) == 0:
            continue
        
        first_count = result.get('first_selected_original_var_count')
        if first_count is not None:
            rank = sum(1 for c in all_counts if c > first_count) + 1
            first_ranks.append(rank)
        
        second_count = result.get('second_selected_original_var_count')
        if second_count is not None:
            rank = sum(1 for c in all_counts if c > second_count) + 1
            second_ranks.append(rank)
    
    if first_selected_percentages:
        stats_lines.append("第一次选择:")
        stats_lines.append(f"  有效选择数: {len(first_selected_percentages)}")
        stats_lines.append(f"  平均出现次数百分比: {np.mean(first_selected_percentages):.4f}%")
        stats_lines.append(f"  中位数出现次数百分比: {np.median(first_selected_percentages):.4f}%")
        stats_lines.append(f"  标准差: {np.std(first_selected_percentages):.4f}%")
        stats_lines.append(f"  最小值: {np.min(first_selected_percentages):.4f}%")
        stats_lines.append(f"  最大值: {np.max(first_selected_percentages):.4f}%")
        stats_lines.append("")
    
    if second_selected_percentages:
        stats_lines.append("第二次选择:")
        stats_lines.append(f"  有效选择数: {len(second_selected_percentages)}")
        stats_lines.append(f"  平均出现次数百分比: {np.mean(second_selected_percentages):.4f}%")
        stats_lines.append(f"  中位数出现次数百分比: {np.median(second_selected_percentages):.4f}%")
        stats_lines.append(f"  标准差: {np.std(second_selected_percentages):.4f}%")
        stats_lines.append(f"  最小值: {np.min(second_selected_percentages):.4f}%")
        stats_lines.append(f"  最大值: {np.max(second_selected_percentages):.4f}%")
        stats_lines.append("")
    
    # 排名分析
    if first_ranks:
        stats_lines.append("第一次选择排名分析 (1=最高频):")
        stats_lines.append(f"  平均排名: {np.mean(first_ranks):.2f}")
        stats_lines.append(f"  中位数排名: {np.median(first_ranks):.2f}")
        stats_lines.append(f"  标准差: {np.std(first_ranks):.2f}")
        top1_ratio = sum(1 for r in first_ranks if r == 1) / len(first_ranks) * 100
        top3_ratio = sum(1 for r in first_ranks if r <= 3) / len(first_ranks) * 100
        top10_ratio = sum(1 for r in first_ranks if r <= 10) / len(first_ranks) * 100
        stats_lines.append(f"  选择最高频变量 (排名=1) 的比例: {top1_ratio:.2f}%")
        stats_lines.append(f"  选择前3高频变量 (排名<=3) 的比例: {top3_ratio:.2f}%")
        stats_lines.append(f"  选择前10高频变量 (排名<=10) 的比例: {top10_ratio:.2f}%")
        stats_lines.append("")
    
    if second_ranks:
        stats_lines.append("第二次选择排名分析 (1=最高频):")
        stats_lines.append(f"  平均排名: {np.mean(second_ranks):.2f}")
        stats_lines.append(f"  中位数排名: {np.median(second_ranks):.2f}")
        stats_lines.append(f"  标准差: {np.std(second_ranks):.2f}")
        top1_ratio = sum(1 for r in second_ranks if r == 1) / len(second_ranks) * 100
        top3_ratio = sum(1 for r in second_ranks if r <= 3) / len(second_ranks) * 100
        top10_ratio = sum(1 for r in second_ranks if r <= 10) / len(second_ranks) * 100
        stats_lines.append(f"  选择最高频变量 (排名=1) 的比例: {top1_ratio:.2f}%")
        stats_lines.append(f"  选择前3高频变量 (排名<=3) 的比例: {top3_ratio:.2f}%")
        stats_lines.append(f"  选择前10高频变量 (排名<=10) 的比例: {top10_ratio:.2f}%")
        stats_lines.append("")
    
    if paired_first and paired_second and len(paired_first) > 1:
        corr = np.corrcoef(paired_first, paired_second)[0, 1]
        stats_lines.append("第一次 vs 第二次选择:")
        stats_lines.append(f"  配对数据数: {len(paired_first)}")
        stats_lines.append(f"  相关系数: {corr:.4f}")
        if len(paired_first) > 3:
            corr_stat, corr_p = stats.pearsonr(paired_first, paired_second)
            stats_lines.append(f"  统计显著性 (p-value): {corr_p:.4f}")
            stats_lines.append(f"  是否显著相关 (p<0.05): {'是 ✓' if corr_p < 0.05 else '否 ✗'}")
        stats_lines.append("")
    
    # 总体结论
    stats_lines.append("="*80)
    stats_lines.append("总体结论")
    stats_lines.append("="*80)
    
    if first_ranks and first_selected_percentages:
        avg_rank_first = np.mean(first_ranks)
        median_rank_first = np.median(first_ranks)
        top3_ratio_first = sum(1 for r in first_ranks if r <= 3) / len(first_ranks)
        avg_percentage_first = np.mean(first_selected_percentages)
        stats_lines.append(f"\n第一次选择:")
        stats_lines.append(f"  - 平均出现次数百分比: {avg_percentage_first:.4f}%")
        stats_lines.append(f"  - 平均排名: {avg_rank_first:.2f}, 中位数排名: {median_rank_first:.1f} (1=最高频)")
        stats_lines.append(f"  - 前3高频比例: {top3_ratio_first*100:.1f}%")
        if median_rank_first < 50:
            stats_lines.append(f"  - 结论: LLM倾向于选择中等偏高频的变量")
        else:
            stats_lines.append(f"  - 结论: LLM的选择与变量频率相关性较弱，倾向于选择中低频变量")
        stats_lines.append("")
    
    if second_ranks and second_selected_percentages:
        avg_rank_second = np.mean(second_ranks)
        median_rank_second = np.median(second_ranks)
        top3_ratio_second = sum(1 for r in second_ranks if r <= 3) / len(second_ranks)
        avg_percentage_second = np.mean(second_selected_percentages)
        stats_lines.append(f"第二次选择:")
        stats_lines.append(f"  - 平均出现次数百分比: {avg_percentage_second:.4f}%")
        stats_lines.append(f"  - 平均排名: {avg_rank_second:.2f}, 中位数排名: {median_rank_second:.1f} (1=最高频)")
        stats_lines.append(f"  - 前3高频比例: {top3_ratio_second*100:.1f}%")
        if median_rank_second < 50:
            stats_lines.append(f"  - 结论: LLM倾向于选择中等偏高频的变量")
        else:
            stats_lines.append(f"  - 结论: LLM的选择与变量频率相关性较弱，倾向于选择中低频变量")
    
    stats_file = os.path.join(output_dir, 'selection_statistics.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(stats_lines))
    logger.info(f"统计摘要已保存到: {stats_file}")


if __name__ == '__main__':
    # 设置随机种子以便复现
    random.seed(42)
    
    # 运行实验
    results = run_experiment()
    
    logger.info("实验脚本执行完成！")

