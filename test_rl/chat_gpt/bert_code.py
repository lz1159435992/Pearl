import sys

from openai import OpenAI
import json

from z3 import parse_smt2_string, Solver

sys.path.append('/home/lz/PycharmProjects/Pearl')
from test_rl.test_script.utils import parse_smt2_in_parts, process_smt_lib_string, fetch_data_as_dict, \
    solve_and_measure_time, model_to_dict, load_dictionary, extract_variables_from_smt2_content, normalize_variables, \
    find_var_declaration_in_string, split_at_check_sat

client = OpenAI(
    # base_url='https://api.openai.com',
    api_key='',
)


def process_text(text):
    # Split the text into chunks of 4096 characters
    responses = []
    # prompts = []
    # prompts.append('我将通过分段的方式给你一个smt文本，你需要对其进行分析，然后为了使其求解加速，给出一个变量和其应该赋值的具体值。')
    # for p in prompts:
    #     chat_completion = client.chat.completions.create(
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": p,
    #             },
    #         ],
    #         model="gpt-4o",
    #     )
    #     # Correct way to get the assistant's message
    #     responses.append(chat_completion.choices[0].message.content)
    chunks = [text[i:i + 4096] for i in range(0, len(text), 4000)]


    for chunk in chunks:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": chunk + '以上是我通过分段的方式给你的smt文本，你需要对其进行分析，然后为了使其求解加速得到sat结果，给出一个或者多个具体的变量名(VAR1,VAR2...)和其应该赋值的具体值。/n',
                },
            ],
            model="gpt-4o",
        )
        # Correct way to get the assistant's message
        responses.append(chat_completion.choices[0].message.content)
    # prompts = []
    # prompts.append(
    #     '根据对上述smt文本的分析，为了使其求解速度变快，给出一个变量和其应该赋值的具体值。')
    # for p in prompts:
    #     chat_completion = client.chat.completions.create(
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": p,
    #             },
    #         ],
    #         model="gpt-4o",
    #     )
    #     # Correct way to get the assistant's message
    #     responses.append(chat_completion.choices[0].message.content)

    return " ".join(responses)

file_path = '/home/lz/baidudisk/smt/gnu_angr.tar.gz/single_test/seq/seq155454'

# file_path = '/home/lz/baidudisk/smt/gnu_angr.tar.gz/single_test/sha1sum/sha1sum77477'
with open(file_path, 'r') as file:
    # 读取文件所有内容到一个字符串
    smtlib_str = file.read()
# 解析字符串
try:
    # 将JSON字符串转换为字典
    dict_obj = json.loads(smtlib_str)
    # print("转换后的字典：", dict_obj)
except json.JSONDecodeError as e:
    print("解析错误：", e)
#
if 'smt-comp' in file_path:
    smtlib_str = dict_obj['smt_script']
else:
    smtlib_str = dict_obj['script']
# variables = set()
variables = extract_variables_from_smt2_content(smtlib_str)
smtlib_str = normalize_variables(smtlib_str, variables)
# long_text = "Hello, how are you?" * 5
#
# response = process_text(smtlib_str)
# print(response)


assertions = parse_smt2_string(smtlib_str)
solver = Solver()
for a in assertions:
    solver.add(a)
timeout = 999999999
# timeout = 1000
result, model, time_taken = solve_and_measure_time(solver, timeout)
print(result,time_taken,model)
# vars_dict = {
#     "VAR1": 9223116941972854571,
#     "VAR2": 11007197058222685532,
#     "VAR3": 1014421176155190799,
#     "VAR4": 1264761117632940449,
#     "VAR5": 5301442144565697567,
#     "VAR6": 15723719345386488079,
#     "VAR7": 2681337752552549496,
#     "VAR8": 11162659525743452955,
#     "VAR9": 13067309850125035466,
#     "VAR10": 9024097345305485713,
#     "VAR11": 11816492885988588662,
#     "VAR12": 4984287078536034086,
#     "VAR13": 5532121768873219521,
#     "VAR14": 12579397499999272079,
#     "VAR15": 3399019372898742203
# }
vars_dict = {
    "VAR21": 21,
    "VAR12": 25,
    "VAR16": 20,
    "VAR1": 30,
    "VAR22": 26,
    "VAR10": 22,
}
# 打印字典以确认
print(vars_dict)
new_constraint_list = []
smtlib_str_before, smtlib_str_after = split_at_check_sat(smtlib_str)
for k,v in vars_dict.items():
    type_info = find_var_declaration_in_string(smtlib_str, k)

    type_scale = type_info.split(' ')[-1]

    new_constraint = "(assert (= {} (_ bv{} {})))\n".format(k, v, type_scale)

    new_constraint_list.append(new_constraint)
new_constraint = ''.join(new_constraint_list)
new_smtlib_str = smtlib_str_before + new_constraint + smtlib_str_after
assertions = parse_smt2_string(new_smtlib_str)
solver = Solver()
for a in assertions:
    solver.add(a)
timeout = 999999999
# timeout = 1000
result, model, time_taken = solve_and_measure_time(solver, timeout)
print(result, time_taken, model)
for k,v in vars_dict.items():
    print(k,v)
    type_info = find_var_declaration_in_string(smtlib_str, k)

    type_scale = type_info.split(' ')[-1]

    new_constraint = "(assert (= {} (_ bv{} {})))\n".format(k, v, type_scale)

    # smtlib_str_before, smtlib_str_after = split_at_check_sat(smtlib_str)

    new_smtlib_str = smtlib_str_before + new_constraint + smtlib_str_after

    assertions = parse_smt2_string(new_smtlib_str)
    solver = Solver()
    for a in assertions:
        solver.add(a)
    timeout = 999999999
    # timeout = 1000
    result, model, time_taken = solve_and_measure_time(solver, timeout)
    print(result, time_taken, model)
#
