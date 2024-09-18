import ast
import json
import random
import re
import time


from test_rl.test_script.utils import parse_smt2_in_parts, process_smt_lib_string, fetch_data_as_dict, \
    solve_and_measure_time, model_to_dict, load_dictionary, extract_variables_from_smt2_content, normalize_variables, \
    normalize_smt_str, find_var_declaration_in_string
from test_rl.test_script.online_learning_break import online_learning

start = time.time()


def test_group():
    count_cons = 0
    count_var = 0
    count_20 = 0
    _20_bit_dict = {}
    with open('/home/lz/sibyl_3/src/networks/info_dict_rl.txt', 'r') as file:
        result_dict = json.load(file)
    items = list(result_dict.items())
    random.shuffle(items)
    result_dict = dict(items)
    var_dict_num = {}
    for key, value in result_dict.items():
        print(key,value)
        list1 = value
        # list1 = ast.literal_eval(value)
        if list1[0] == "sat":
            if list1[1] > 20 and 'sort21837' not in key:
                count_20 += 1
                # if '/who/who86404' in key:
                print(key, value)
                if '/home/yy/Downloads/' in key:
                    file_path = key.replace('/home/yy/Downloads/', '/home/lz/baidudisk/')
                elif '/home/nju/Downloads/' in key:
                    file_path = key.replace('/home/nju/Downloads/', '/home/lz/baidudisk/')
                else:
                    file_path = key
                    # if 'gnu_angr.tar.gz/single_test/cat/cat43772' in file_path:
                    #     continue
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
                variables = set()
                # variables = extract_variables_from_smt2_content(smtlib_str)
                # smtlib_str = normalize_variables(smtlib_str, variables)
                smtlib_str, var_dict, constant_list = normalize_smt_str(smtlib_str)
                count_cons_flag = False
                tmp = 0
                for k,v in var_dict.items():
                    print(k,v)
                    type_info = find_var_declaration_in_string(smtlib_str, v)
                    if type_info:
                        print(type_info)
                        print(type(type_info))
                        type_scale = type_info.split(' ')[-1]
                        print(type_scale)
                        #判断最大的位数
                        if int(type_scale) > tmp:
                            tmp = int(type_scale)

                        if 10 ** int(type_scale) >= 10 ** 100:
                            count_cons_flag = True
                            count_var += 1
                        if type_scale not in var_dict_num.keys():
                            var_dict_num[type_scale] = 1
                        else:
                            var_dict_num[type_scale] += 1
                    if count_cons_flag:
                        count_cons += 1
                # 对位数进行分类
                if tmp not in _20_bit_dict.keys():
                    _20_bit_dict[tmp] = []
                    _20_bit_dict[tmp].append(file_path)
                else:
                    _20_bit_dict[tmp].append(file_path)

    print(count_cons,count_var)
    print(len(result_dict))
    print(count_20)
    print(var_dict_num)
    with open('info_bit_dict.txt', 'w') as file:
        json.dump(_20_bit_dict, file, indent=4)
if __name__ == '__main__':
    test_group()
    # a = 2 **32
    # print(a)