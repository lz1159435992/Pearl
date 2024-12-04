import json
import re
# str = '[  {"VAR1": 5, "VAR2": 9},{"VAR3": 87, "VAR4": 96}]'
# dict_obj = json.loads(str)
# print(dict_obj[0])
# print(type(dict_obj[0]))


action_str ='''
To analyze the provided SMT-LIB file and extract the relevant information, let\'s proceed step by step.\n\n### 1. Extract and List All Variables and Functions\nThe SMT-LIB file declares the following functions (variables):\n```json\n{\n  "variables": [\n    "VAR1",\n    "VAR2",\n    "VAR3",\n    "VAR4",\n    "VAR5",\n    "VAR6",\n    "VAR7",\n    "VAR8",\n    "VAR9",\n    "VAR10",\n    "VAR11",\n    "VAR12",\n    "VAR13",\n    "VAR14",\n    "VAR15",\n    "VAR16",\n    "VAR17"\n  ]\n}\n```\n\n### 2. Parse and Explain Logical Expressions and Constraints\nThe assertions (`assert` statements) define constraints on these variables, often using bit-wise operations and comparisons. Here\'s a brief overview of the key logical expressions:\n\n- **Variable Equality Constraints:** \n  - Most `assert` statements involve setting a variable to a value derived from others, often structured as:\n    ``` \n    (not (= VARx derived_value))\n    ```\n  - Each variable `VARx` is negated in the context of equality to its derived value, meaning it must not be equal.\n\n- **Arithmetic Operations:**\n  The expressions frequently use operations such as `bvadd` (bit-vector addition) and `bvmul` (bit-vector multiplication) to compute values based on other variables.\n\n- **Distinctness Constraints:**\n  The constraints also include checks for distinctness against specific bit-vector values.\n\n- **Final Variable Checks:**\n  At the end, `VAR17` must equal zero, and there\'s a comprehensive summation of variables to check if the sum respects certain bounds.\n\n### 3. Analyze SMT Content for Variable Assignments\nGiven the constraints, to satisfy all conditions, we need to select values for the variables that:\n- Avoids producing values that the `not` clauses negate.\n- Satisfies any distinctness conditions.\n\n### 4. Provide a Summary of the Simplified Constraints\nTo simplify the constraints, we can start with a base case:\n- Set the bit-vector values of the variables to zero initially, as many constraints require them not to equal zero (by negating the checks).\n- Gradually introduce values for each variable until we find a combination that satisfies all assertions.\n\n### 5. Select 10 Sets of Variables and Specific values\nThe following are proposed variable assignments that satisfy the constraints based on the analysis above:\n\n```json\n[\n  {"VAR1": 0, "VAR2": 1, "VAR3": 2, "VAR4": 3, "VAR5": 4, "VAR6": 5, "VAR7": 6, "VAR8": 7, "VAR9": 8, "VAR10": 9},\n  {"VAR1": 0, "VAR2": 2, "VAR3": 4, "VAR4": 6, "VAR5": 8, "VAR6": 10, "VAR7": 12, "VAR8": 14, "VAR9": 16, "VAR10": 18},\n  {"VAR1": 1, "VAR2": 3, "VAR3": 5, "VAR4": 7, "VAR5": 9, "VAR6": 11, "VAR7": 13, "VAR8": 15, "VAR9": 17, "VAR10": 19},\n  {"VAR1": 7, "VAR2": 8, "VAR3": 9, "VAR4": 10, "VAR5": 11, "VAR6": 12, "VAR7": 13, "VAR8": 14, "VAR9": 15, "VAR10": 16},\n  {"VAR1": 3, "VAR2": 6, "VAR3": 9, "VAR4": 12, "VAR5": 15, "VAR6": 18, "VAR7": 21, "VAR8": 24, "VAR9": 27, "VAR10": 30},\n  {"VAR1": 10, "VAR2": 20, "VAR3": 30, "VAR4": 40, "VAR5": 50, "VAR6": 60, "VAR7": 70, "VAR8": 80, "VAR9": 90, "VAR10": 100},\n  {"VAR1": 15, "VAR2": 30, "VAR3": 45, "VAR4": 60, "VAR5": 75, "VAR6": 90, "VAR7": 105, "VAR8": 120, "VAR9": 135, "VAR10": 150},\n  {"VAR1": 5, "VAR2": 10, "VAR3": 15, "VAR4": 20, "VAR5": 25, "VAR6": 30, "VAR7": 35, "VAR8": 40, "VAR9": 45, "VAR10": 50},\n  {"VAR1": 9, "VAR2": 18, "VAR3": 27, "VAR4": 36, "VAR5": 45, "VAR6": 54, "VAR7": 63, "VAR8": 72, "VAR9": 81, "VAR10": 90},\n  {"VAR1": 2, "VAR2": 4, "VAR3": 6, "VAR4": 8, "VAR5": 10, "VAR6": 12, "VAR7": 14, "VAR8": 16, "VAR9": 18, "VAR10": 20}\n]\n```\n\nThis approach evaluates simple values to avoid conflict with the constraints. Adjustments might be needed based on the actual results of each assignment when checked against the assertions in the SMT-LIB file.
'''
json_pattern = r'```json\n(.*?)```'
json_matches = re.findall(json_pattern, action_str, re.DOTALL)

parsed_jsons = []
for json_string in json_matches:
    try:
        # 尝试解析JSON字符串
        parsed_json = json.loads(json_string)
        parsed_jsons.append(parsed_json)
        print("JSON提取并解析成功：")
        print(json.dumps(parsed_json, indent=4, ensure_ascii=False))
    except json.JSONDecodeError as e:
        print("JSON解析失败：", e)

# 打印所有解析的JSON数据
for parsed_json in parsed_jsons:
    print(parsed_json)
print(parsed_jsons[-1])
# print(json_string)
# print(parsed_json)