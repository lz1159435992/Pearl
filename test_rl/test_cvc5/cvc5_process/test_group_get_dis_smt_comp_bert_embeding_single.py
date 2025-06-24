import ast
import json
import os
import random


import time
import tqdm

from z3.z3 import is_const
from z3.z3consts import Z3_OP_UNINTERPRETED

from pearl.SMTimer.KNN_Predictor import Predictor


import torch

import numpy as np

# from bert_embedder import CodeEmbedder_normalize
from test_rl.predictor.bert_embedder_test import CodeEmbedder_normalize

from test_rl.test_script.utils import parse_smt2_in_parts, process_smt_lib_string, fetch_data_as_dict, \
    solve_and_measure_time, model_to_dict, load_dictionary, extract_variables_from_smt2_content, normalize_variables, \
    normalize_smt_str, normalize_smt_str_without_replace

start = time.time()


# def extract_variables_from_smt2_content(content):
#     """
#     从 SMT2 格式的字符串内容中提取变量名。
#
#     参数:
#     - content: SMT2 格式的字符串内容。
#
#     返回:
#     - 变量名列表。
#     """
#     # 用于匹配 `(declare-fun ...)` 语句中的变量名的正则表达式
#     variable_pattern = re.compile(r'\(declare-fun\s+([^ ]+)')
#
#     # 存储提取的变量名
#     variables = []
#
#     # 按行分割字符串并迭代每一行
#     for line in content.splitlines():
#         # 在每一行中查找匹配的变量名
#         match = variable_pattern.search(line)
#         if match:
#             # 如果找到匹配项，则将变量名添加到列表中
#             variables.append(match.group(1).replace('|', ''))
#
#     return set(variables)


def visit(expr):
    if is_const(expr) and expr.decl().kind() == Z3_OP_UNINTERPRETED:
        # Add only uninterpreted functions (which represent variables)
        variables.add(str(expr))
    else:
        # Recursively visit children for composite expressions
        for child in expr.children():
            visit(child)




def test_group_bert_normalize(file_path='result_dict_time_pre_order.txt'):
    with open(file_path, 'r') as file:
        result_dict = json.load(file)

    embedder = CodeEmbedder_normalize()
    # 遍历字典并统计数据
    features_list = []
    for (i, (key, value)) in enumerate(tqdm.tqdm(result_dict.items())):
        if True:
        # if 'sort29776' in key:
            file_path = key
            # print(key)
            with open(file_path, 'r') as file:
                # 读取文件所有内容到一个字符串
                smtlib_str = file.read()
                file.close()
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
            print(file_path)

            var_dict = normalize_smt_str_without_replace(smtlib_str)

            smtlib_str, var_dict, constant_list = normalize_smt_str(smtlib_str)
            v_list = extract_variables_from_smt2_content(smtlib_str)
            print('变量列表：')
            print(v_list)
            smtlib_str, var_dict, constant_list = normalize_smt_str(smtlib_str)
            # print('规范化后的字符串：')
            # print(smtlib_str)
            var_dict = normalize_smt_str_without_replace(smtlib_str)
            # print('规范化后的字符串：')
            # print(smtlib_str)
            print('变量字典：')
            print(var_dict)
            # print('常量列表：')
            # print(constant_list)
            embeddings = embedder.get_max_pooling_embedding(smtlib_str, var_dict)
    # 进行测试
    #     features_list.append(embeddings)
    #
    # features_array = np.array([t.numpy() if isinstance(t, torch.Tensor) else t for t in features_list])
    # # 保存特征和标签数组到文件
    # np.save('features_normal.npy', features_array)

    # with open('result_dict_time_pre_order.txt', 'w') as file:
    #     # 使用json.dump()将字典保存到文件
    #     json.dump(result_dict, file, indent=4)


def test_group_bert_normalize_1by1(file_path='result_dict_time_pre_order.txt'):
    with open(file_path, 'r') as file:
        result_dict = json.load(file)

    embedder = CodeEmbedder_normalize()

    # 遍历字典并统计数据
    for (i, (file_path, _)) in enumerate(tqdm.tqdm(result_dict.items())):
        # 读取文件内容
        with open(file_path, 'r') as file:
            smtlib_str = file.read()
            file.close()
        # 解析字符串为JSON
        try:
            dict_obj = json.loads(smtlib_str)
            if 'smt-comp' in file_path:
                smtlib_str = dict_obj['smt_script']
            else:
                smtlib_str = dict_obj['script']
        except json.JSONDecodeError as e:
            print("解析错误：", e)
            continue  # 如果解析出错，跳过当前文件，继续处理下一个文件

        # 处理smtlib_str
        smtlib_str, var_dict,_ = normalize_smt_str(smtlib_str)
        print(smtlib_str)
        embeddings = embedder.get_max_pooling_embedding(smtlib_str, var_dict)

        # 将embeddings转换为numpy数组，如果它是一个torch.Tensor
        embeddings_array = embeddings.detach().numpy() if isinstance(embeddings, torch.Tensor) else np.array(embeddings)

        # 保存当前文件的特征到单独的Numpy文件中
        np.save(f'features/features_normal_{i}.npy', embeddings_array)
        del embeddings_array
        # 如果需要的话，可以在这里清空 embeddings_array 以节省内存

def test_group_bert_normalize_1by1_smt():
    import pysmt.logics

    with open('smt_v2.json', 'r') as file:
        result_dict = json.load(file)
    features_list = []
    # labels_list = []
    time_list = []
    embedder = CodeEmbedder_normalize()
    logic_systems = [
        "QF_BOOL", "QF_IDL", "QF_LIA", "QF_LRA", "QF_RDL", "QF_UF", "QF_UFIDL",
        "QF_UFLIA", "QF_UFLRA", "QF_UFLIRA",
        "BOOL", "LRA", "LIA", "UFLIRA", "UFLRA",
        "QF_BV", "QF_UFBV",
        "QF_SLIA",
        "QF_BV", "QF_UFBV",
        "QF_ABV", "QF_AUFBV", "QF_AUFLIA", "QF_ALIA", "QF_AX",
        "QF_AUFBVLIRA",
        "QF_NRA", "QF_NIA", "UFBV", "BV"
    ]
    # 遍历字典并统计数据
    for (i, (file_path,v)) in enumerate(tqdm.tqdm(result_dict.items())):
        # 读取文件内容
        if len(v) == 0 or file_path.split('/')[6] not in logic_systems:
            continue
        with open(file_path, 'r') as file:
            smtlib_str = file.read()
            file.close()

        # 处理smtlib_str
        try:
            start_time = time.time()
            smtlib_str, var_dict,_ = normalize_smt_str(smtlib_str)
            end_time = time.time()
            if end_time - start_time > 30:
                print(file_path)
        except Exception as e:
            print(e)
            result_dict[file_path].append('解析错误')
            continue
        # 处理异常

        print(smtlib_str)
        embeddings = embedder.get_max_pooling_embedding(smtlib_str, var_dict)
        features_list.append(embeddings)
        time_value = float(v[2])
        if time_value <= 1:
            time_list.append(0)
        elif time_value <= 20:
            time_list.append(1)
        elif time_value <= 50:
            time_list.append(2)
        elif time_value <= 100:
            time_list.append(3)
        elif time_value <= 200:
            time_list.append(4)
        elif time_value <= 500:
            time_list.append(5)
        elif time_value <= 1200:
            time_list.append(6)
        else:
            time_list.append(7)
        # # 将embeddings转换为numpy数组，如果它是一个torch.Tensor
        # embeddings_array = embeddings.detach().numpy() if isinstance(embeddings, torch.Tensor) else np.array(embeddings)
        # # 保存当前文件的特征到单独的Numpy文件中
        # np.save(f'smt_comp_features/features_normal_{i}.npy', embeddings_array)

        features_array = np.array([t.numpy() if isinstance(t, torch.Tensor) else t for t in features_list])
        # labels_array = np.array(labels_list)
        time_array = np.array(time_list)
        # 保存特征和标签数组到文件
    #     np.save('smt_comp_features/features.npy', features_array)
    #     # np.save('labels.npy', labels_array)
    #     np.save('smt_comp_features/time.npy', time_array)
    #     # del embeddings_array
    #     # 如果需要的话，可以在这里清空 embeddings_array 以节省内存
    # with open('smt_v3.json', 'w', encoding='utf-8') as file:
    #     json.dump(result_dict, file, ensure_ascii=False, indent=4)
def test_group_bert_normalize_1by1_smt_name():
    import pysmt.logics

    with open('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_smt_comp.txt', 'r') as file:
        result_dict = json.load(file)
    embeding_dict = {}
    features_list = []
    # labels_list = []
    time_list = []
    embedder = CodeEmbedder_normalize()
    logic_systems = [
        # "QF_BOOL",
        # "QF_IDL",
        "QF_LIA",
        # "QF_LRA",
        # "QF_RDL",
        # "QF_UF",
        # "QF_UFIDL",
        # # "QF_UFLIA", "QF_UFLRA", "QF_UFLIRA",
        # # "BOOL",
        # "LRA", "LIA",
        # # "UFLIRA", "UFLRA",
        # "QF_BV",
        # "QF_UFBV",
        # "QF_SLIA",
        # "QF_BV",
        # "QF_UFBV",
        # # "QF_ABV", "QF_AUFBV", "QF_AUFLIA", "QF_ALIA", "QF_AX",
        # # "QF_AUFBVLIRA",
        # "QF_NRA",
        # "QF_NIA",
        # # "UFBV", "BV"
    ]
    output_path = '/home/lz/PycharmProjects/Pearl/test_rl/predictor/smt_comp_features'
    # 遍历字典并统计数据
    for (i, (file_path, v)) in enumerate(tqdm.tqdm(result_dict.items())):
        print(file_path, v)
        # 读取文件内容
        if file_path.split('/')[6] not in logic_systems:
            continue
        output_path_file = os.path.join(output_path, file_path.split('/')[6]+'1000')
        if not os.path.exists(output_path_file):
            # 如果目录不存在，则创建目录
            os.makedirs(output_path_file)
        with open(file_path, 'r') as file:
            smtlib_str = file.read()
            file.close()
        # 处理smtlib_str
        len_var = extract_variables_from_smt2_content(smtlib_str)
        print(len(len_var))
        if len(len_var) > 1000:
            continue
        # 处理smtlib_str
        try:
            smtlib_str, var_dict, _ = normalize_smt_str(smtlib_str)
            embeding_dict[file_path] = []
        except Exception as e:
            print(e)
            embeding_dict[file_path].append('解析错误')
            continue
        # 处理异常

        # print(smtlib_str)
        embeddings = embedder.get_max_pooling_embedding(smtlib_str, var_dict)

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().numpy()
        np.save(os.path.join(output_path_file, file_path.replace("/", "_")+'.npy'), embeddings)
        embeding_dict[file_path].append(os.path.join(output_path_file, file_path.replace("/", "_")+'.npy'))

        # features_list.append(embeddings)

        time_value = float(v[1])
        print(time_value)
        if time_value <= 1:
            embeding_dict[file_path].append(0)
        elif time_value <= 20:
            embeding_dict[file_path].append(1)
        elif time_value <= 50:
            embeding_dict[file_path].append(2)
        elif time_value <= 100:
            embeding_dict[file_path].append(3)
        elif time_value <= 200:
            embeding_dict[file_path].append(4)
        elif time_value <= 500:
            embeding_dict[file_path].append(5)
        elif time_value <= 1200:
            embeding_dict[file_path].append(6)
        else:
            embeding_dict[file_path].append(7)
        # # 将embeddings转换为numpy数组，如果它是一个torch.Tensor
        # embeddings_array = embeddings.detach().numpy() if isinstance(embeddings, torch.Tensor) else np.array(embeddings)
        # # 保存当前文件的特征到单独的Numpy文件中
        # np.save(f'smt_comp_features/features_normal_{i}.npy', embeddings_array)

        # features_array = np.array([t.numpy() if isinstance(t, torch.Tensor) else t for t in features_list])
        # # labels_array = np.array(labels_list)
        # time_array = np.array(time_list)
        # 保存特征和标签数组到文件

        # np.save('labels.npy', labels_array)
        # np.save(os.path.join(output_path_file, 'time.npy'), time_array)
        # del embeddings_array
        # 如果需要的话，可以在这里清空 embeddings_array 以节省内存
        with open('embeding_QF_LIA.json', 'w', encoding='utf-8') as file:
            json.dump(embeding_dict, file, ensure_ascii=False, indent=4)

def convert_timeout_to_unknown(solve_dict):
    """
    将solve_dict中每个value的第一个值如果是'timeout'则改为'unknown'
    
    Args:
        solve_dict: 原始字典
    
    Returns:
        修改后的字典
    """
    modified_dict = {}
    for key, value in solve_dict.items():
        if value[0] == 'timeout':
            # 创建新的value列表，第一个元素改为'unknown'，其他保持不变
            new_value = ['unknown'] + value[1:]
            modified_dict[key] = new_value
        else:
            modified_dict[key] = value
    return modified_dict

def test_group_get_label_and_time(file_path='result_dict_time_pre.txt',solve_path='/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/cvc5_smtimer_results_rl.json'):
    with open(file_path, 'r') as file:
        result_dict = json.load(file)
    with open(solve_path, 'r') as file:
        solve_dict = json.load(file)
        
    # 转换timeout为unknown
    solve_dict = convert_timeout_to_unknown(solve_dict)
        
    # 创建新的字典来存储替换后的路径
    # new_result_dict = {}
    # for key in result_dict:
    #     if '/home/lz/baidudisk/' in key:
    #         new_key = key.replace('/home/lz/baidudisk/', '/home/nju/Downloads/')
    #         new_result_dict[new_key] = result_dict[key]
    #     else:
    #         new_result_dict[key] = result_dict[key]
    # result_dict = new_result_dict
    #
    # # 同样替换solve_dict中的路径
    # new_solve_dict = {}
    # for key in solve_dict:
    #     if '/home/lz/baidudisk/' in key:
    #         new_key = key.replace('/home/lz/baidudisk/', '/home/nju/Downloads/')
    #         new_solve_dict[new_key] = solve_dict[key]
    #     else:
    #         new_solve_dict[key] = solve_dict[key]
    # solve_dict = new_solve_dict

    stats = {
        'sat': {'count': 0, 'percentage': 0},
        'unsat': {'count': 0, 'percentage': 0},
        'unknown': {'count': 0, 'percentage': 0},
        'times': {
            '1': {'count': 0, 'percentage': 0},
            '20': {'count': 0, 'percentage': 0},
            '50': {'count': 0, 'percentage': 0},
            '100': {'count': 0, 'percentage': 0},
            '200': {'count': 0, 'percentage': 0},
            '500': {'count': 0, 'percentage': 0}
        }
    }

    # 遍历字典并统计数据
    total_entries = len(result_dict)

    labels_list = []
    time_list = []
    for (i, (key, value)) in enumerate(tqdm.tqdm(result_dict.items())):
        category = solve_dict[key][0]  # "sat", "unsat", 或 "unknown"
        time_value = solve_dict[key][1]  # 消耗的时间
        result_dict[key].append(i)
        if solve_dict[key][0] == "sat":
            labels_list.append(0)
            if time_value <= 1:
                time_list.append(1)
                result_dict[key].append(1)
            elif time_value <= 20:
                time_list.append(2)
                result_dict[key].append(2)
            elif time_value <= 50:
                time_list.append(3)
                result_dict[key].append(3)
            elif time_value <= 100:
                time_list.append(4)
                result_dict[key].append(4)
            elif time_value <= 200:
                time_list.append(5)
                result_dict[key].append(5)
            elif time_value <= 500:
                time_list.append(6)
                result_dict[key].append(6)
            else:
                time_list.append(7)
                result_dict[key].append(7)
        else:
            labels_list.append(1)
            # 没有时间
            time_list.append(0)
            result_dict[key].append(0)

        # 更新类别统计
        stats[category]['count'] += 1

        # 更新时间统计
        if time_value <= 1:
            time_key = '1'
        elif time_value <= 20:
            time_key = '20'
        elif time_value <= 50:
            time_key = '50'
        elif time_value <= 100:
            time_key = '100'
        elif time_value <= 200:
            time_key = '200'
        elif time_value <= 500:
            time_key = '500'
        else:
            time_key = '500'  # 超过500的时间归类到500
        stats['times'][time_key]['count'] += 1
    # 计算百分比
    for category in stats:
        if category != 'times':
            stats[category]['percentage'] = (stats[category]['count'] / total_entries) * 100

    for time_key in stats['times']:
        stats['times'][time_key]['percentage'] = (stats['times'][time_key]['count'] / total_entries) * 100

    # 打印统计结果
    print("Category Statistics:")
    for category, data in stats.items():
        if category != 'times':
            print(f"  {category}: Count = {data['count']}, Percentage = {data['percentage']:.2f}%")

    print("\nTime Statistics:")
    for time_key, data in stats['times'].items():
        print(f"  Time <= {time_key}: Count = {data['count']}, Percentage = {data['percentage']:.2f}%")

    labels_array = np.array(labels_list)
    time_array = np.array(time_list)
    # 保存特征和标签数组到文件
    np.save('labels.npy', labels_array)
    np.save('time.npy', time_array)

def run_complete_process(info_dict_path, solve_dict_path, features_dir='features', model_save_dir='models'):
    """
    运行完整的数据处理和模型训练流程
    
    Args:
        info_dict_path: 信息字典文件路径
        solve_dict_path: 求解结果字典文件路径
        features_dir: 特征文件保存目录
        model_save_dir: 模型保存目录
    """
    import os
    
    # 创建必要的目录
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    
    print("=== 第一步：处理数据集，生成特征和标签 ===")
    # 处理数据并生成标签
    test_group_get_label_and_time(info_dict_path, solve_dict_path)
    print("特征和标签生成完成！")
    
    print("\n=== 第二步：训练二分类模型 ===")
    from train_predictor import train_binary_classifier
    binary_save_path = os.path.join(model_save_dir, 'binary_classifier.pth')
    train_binary_classifier(
        features_dir=features_dir,
        labels_path='labels.npy',
        save_path=binary_save_path
    )
    print(f"二分类模型已保存到：{binary_save_path}")
    
    print("\n=== 第三步：训练八分类模型 ===")
    from train_predictor import train_eight_class_model
    eight_class_save_path = os.path.join(model_save_dir, 'eight_class_model.pth')
    train_eight_class_model(
        features_dir=features_dir,
        time_labels_path='time.npy',
        save_path=eight_class_save_path
    )
    print(f"八分类模型已保存到：{eight_class_save_path}")
    
    print("\n=== 所有处理完成！===")
    print(f"- 特征文件保存在：{features_dir}/")
    print(f"- 标签文件：labels.npy 和 time.npy")
    print(f"- 模型文件保存在：{model_save_dir}/")

if __name__ == '__main__':
    # 示例：运行完整处理流程
    run_complete_process(
        info_dict_path='/home/lz/constraint_solve_file/info_dict_predictor.txt',
        solve_dict_path='/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/cvc5_smtimer_results_predictor.json',
        features_dir='/home/lz/PycharmProjects/Pearl/test_rl/features',
        model_save_dir='models'
    )
    
    # 或者单独运行数据处理
    # test_group_get_label_and_time('/home/nju/constraint_solve_file/info_dict_predictor.txt',
    #                               '/home/nju/PycharmProjects/Pearl/test_rl/test_cvc5/smtimer_710/mathsat5_smtimer_results_predictor.json')

    """
    完整执行示例：

    1. 在Python代码中运行完整流程：
    from test_group_get_dis_smt_comp_bert_embeding_single import run_complete_process
    
    run_complete_process(
        info_dict_path='/path/to/info_dict_predictor.txt',
        solve_dict_path='/path/to/mathsat5_smtimer_results_predictor.json',
        features_dir='features',
        model_save_dir='models'
    )

    2. 或通过命令行分步执行：

    # 处理数据集，生成特征和标签：
    python test_group_get_dis_smt_comp_bert_embeding_single.py

    # 训练二分类模型：
    python train_predictor.py --model_type binary --features_dir features/ --save_path models/binary_classifier.pth

    # 训练八分类模型：
    python train_predictor.py --model_type eight_class --features_dir features/ --save_path models/eight_class_model.pth

    注意事项：
    1. 确保数据文件路径正确
    2. 特征文件会保存在features/目录下
    3. 模型文件会保存在models/目录下
    4. 执行前确保已安装所有必要的Python包：
       - torch
       - numpy
       - tqdm
    """

    # loaded_features = np.load('features.npy')
    # loaded_labels = np.load('labels.npy')
    # print(loaded_features.shape)
