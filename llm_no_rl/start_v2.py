import os
import sys

from openai import OpenAI
import json

from z3 import parse_smt2_string, Solver
os.environ['ALL_PROXY'] = ''
os.environ['all_proxy'] = ''
sys.path.append('/home/lz/PycharmProjects/Pearl')
from test_rl.test_script.utils import parse_smt2_in_parts, process_smt_lib_string, fetch_data_as_dict, \
    solve_and_measure_time, model_to_dict, load_dictionary, extract_variables_from_smt2_content, normalize_variables, \
    find_var_declaration_in_string, split_at_check_sat

client = OpenAI(
    base_url='https://svip.xty.app/v1',
    api_key=''

)


def process_text(text):
    # Split the text into chunks of 4096 characters
    text_limit = 120000
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
    chunks = [text[i:i + text_limit] for i in range(0, len(text), text_limit)]


    # Add 'system' role message before the loop
    system_message = {
        "role": "system",
        "content": "You are an advanced SAT/SMT solver, focusing on the optimization and resolution of logical constraint problems. Your input consists of two parts: first, the counterexamples of failed solution assignments previously chosen in json formats, and second, the strings in SMT-LIB format that needs to be solved. the above two parts are divided by -------------------------------------. You should analyze these inputs, using logical reasoning and heuristic methods to determine which variable assignments led to the failure of the solution, and identify the variable assignments that satisfy all constraint conditions. The output should be 10 sets of specific variables assignments that can satisfy all the constraints defined in the SMT-LIB strings. Your task is to select 10 sets of variables and their corresponding specific values from the SMT file so that the SMT file can be solved as satisfiable (SAT). One set of variables can include all the variables in the constraints, or it can contain only some of the variables. Your output should conform to the JSON format, with an example as follows: \n [  {<variable1>: <value1>, <variable2>: <value2>,...},{<variable1>: <value3>, <variable3>: <value4>,...} ...]"
    }


    # '以上是我通过分段的方式给你的smt文本，你需要对其进行分析，然后为了使其求解加速得到sat结果，给出一个或者多个具体的变量名(VAR1,VAR2...)和其应该赋值的具体值。/n',
    for chunk in chunks:
        chat_completion = client.chat.completions.create(
            messages=[
                system_message,  # Including the system role message here
                {
                    "role": "user",
                    "content": chunk + f'This is the The variable values from the previous failed SAT solving attempt and SMT text given to you in segments; analyze it. To speed up the solution and obtain a SAT result, and identify the variable assignments that satisfy all constraint conditions. The output should be 10 sets of specific variables assignments that can satisfy all the constraints defined in the SMT-LIB strings. Your task is to select 10 sets of variables and their corresponding specific values from the SMT file so that the SMT file can be solved as satisfiable (SAT). Your output should conform to the JSON format. Do not output any other text, explanations.'
                },
            ],
            # model="gpt-3.5-turbo",
            model="gpt-4o-mini",
            # max_tokens = 10,
            temperature = 0.7,

        )
        # Correct way to get the assistant's message
        # print(chat_completion.choices[0].message.content)
        responses.append(chat_completion.choices[0].message.content)

    print(responses)
    return responses[-1]

# file_path = '/home/lz/baidudisk/smt/gnu_angr.tar.gz/single_test/seq/seq155454'
# file_path = '/home/lz/baidudisk/smt/gnu_angr.tar.gz/single_test/seq/seq155454'
file_path = '/home/lz/baidudisk/smt/gnu_angr.tar.gz/single_test/sha1sum/sha1sum77477'
# file_path = '/home/lz/baidudisk/smt/buzybox_angr.tar.gz/single_test/udhcpc/udhcpc6664876'
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
# smtlib_str = "Hello, how are you?" * 5
response = process_text(smtlib_str)
with open('example.txt', 'w', encoding='utf-8') as file:
    # 将字符串写入文件
    file.write(response)
print(response)


# assertions = parse_smt2_string(smtlib_str)
# solver = Solver()
# for a in assertions:
#     solver.add(a)
# timeout = 999999999
# # timeout = 1000
# result, model, time_taken = solve_and_measure_time(solver, timeout)
# print(result,time_taken,model)
# # vars_dict = {
# #     "VAR1": 9223116941972854571,
# #     "VAR2": 11007197058222685532,
# #     "VAR3": 1014421176155190799,
# #     "VAR4": 1264761117632940449,
# #     "VAR5": 5301442144565697567,
# #     "VAR6": 15723719345386488079,
# #     "VAR7": 2681337752552549496,
# #     "VAR8": 11162659525743452955,
# #     "VAR9": 13067309850125035466,
# #     "VAR10": 9024097345305485713,
# #     "VAR11": 11816492885988588662,
# #     "VAR12": 4984287078536034086,
# #     "VAR13": 5532121768873219521,
# #     "VAR14": 12579397499999272079,
# #     "VAR15": 3399019372898742203
# # }
# vars_dict = {
#     "VAR21": response
# }
# # # 打印字典以确认
# # print(vars_dict)
# new_constraint_list = []
# smtlib_str_before, smtlib_str_after = split_at_check_sat(smtlib_str)
# for k,v in vars_dict.items():
#     type_info = find_var_declaration_in_string(smtlib_str, k)
#
#     type_scale = type_info.split(' ')[-1]
#
#     new_constraint = "(assert (= {} (_ bv{} {})))\n".format(k, v, type_scale)
#
#     new_constraint_list.append(new_constraint)
# new_constraint = ''.join(new_constraint_list)
# new_smtlib_str = smtlib_str_before + new_constraint + smtlib_str_after
# assertions = parse_smt2_string(new_smtlib_str)
# solver = Solver()
# for a in assertions:
#     solver.add(a)
# timeout = 999999999
# # timeout = 1000
# result, model, time_taken = solve_and_measure_time(solver, timeout)
# print(result, time_taken, model)
# for k,v in vars_dict.items():
#     print(k,v)
#     type_info = find_var_declaration_in_string(smtlib_str, k)
#
#     type_scale = type_info.split(' ')[-1]
#
#     new_constraint = "(assert (= {} (_ bv{} {})))\n".format(k, v, type_scale)
#
#     # smtlib_str_before, smtlib_str_after = split_at_check_sat(smtlib_str)
#
#     new_smtlib_str = smtlib_str_before + new_constraint + smtlib_str_after
#
#     assertions = parse_smt2_string(new_smtlib_str)
#     solver = Solver()
#     for a in assertions:
#         solver.add(a)
#     timeout = 999999999
#     # timeout = 1000
#     result, model, time_taken = solve_and_measure_time(solver, timeout)
#     print(result, time_taken, model)
# #
