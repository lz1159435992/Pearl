"""
QF_NIA数据处理和Embedding生成模块
基于test_group_get_dis_smt_comp_llm.py中的方法实现
用于处理QF_NIA问题集并生成LLM embeddings
"""
import ast
import json
import os
import sys
import signal
import random
import time
import tqdm

os.environ['ALL_PROXY'] = ''
os.environ['all_proxy'] = ''

import torch
import numpy as np
from ollama import Client
from loguru import logger

# 添加项目路径
sys.path.append('/home/lz/PycharmProjects/Pearl')
from test_rl.bert_embedder_test import CodeEmbedder_normalize
from test_rl.test_script.utils import (
    extract_variables_from_smt2_content, 
    normalize_smt_str, 
    MyException, 
    timeout_handler,
    setup_logger
)

# 设置日志

def process_embeding(text, llm_host='http://172.29.7.221:32827', llm_model='llama3.1:70b'):
    """
    使用LLM生成文本的embedding
    
    Args:
        text: 输入的SMT-LIB文本
        llm_host: LLM服务器地址
        llm_model: 使用的LLM模型
        
    Returns:
        torch.Tensor: embedding向量
    """
    client = Client(host=llm_host)
    response = client.embeddings(
        model=llm_model,
        prompt=text,
        options={"temperature": 0},
    )
    logger.info(f'Embedding shape: {len(response["embedding"])}')
    return torch.tensor(response['embedding'])

def convert_timeout_to_unknown(result_dict):
    """
    将timeout状态转换为unknown
    
    Args:
        result_dict: 求解结果字典
        
    Returns:
        dict: 处理后的结果字典
    """
    for key, value in result_dict.items():
        if isinstance(value, list) and len(value) > 0:
            if value[0] == 'timeout':
                value[0] = 'unknown'
    return result_dict

def test_group_bert_normalize_1by1_smt_name_2_QF_NIA():
    """
    处理QF_NIA问题集并生成embedding
    基于test_group_get_dis_smt_comp_llm_time_out.py中的test_group_bert_normalize_1by1_smt_name_2方法
    """
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 读取配置文件
    config_path = os.path.join(script_dir, 'config.json')
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    # 获取LLM配置
    llm_host = config['embedding']['host']
    llm_model = config['embedding']['model']
    logger.info(f'LLM配置 - Host: {llm_host}, Model: {llm_model}')
    
    # 读取QF_NIA求解结果
    with open('/home/lz/PycharmProjects/Pearl/test_rl/test_QF_NIA/cvc5_QF_NIA.json', 'r') as file:
        result_dict = json.load(file)
    
    # 转换timeout为unknown
    result_dict = convert_timeout_to_unknown(result_dict)
    
    # 加载全局已有的embedding字典（从predictor目录）
    global_embedding_file = '/home/lz/PycharmProjects/Pearl/test_rl/predictor/smt_comp_NIA/embeding_QF_NIA.json'
    if os.path.exists(global_embedding_file):
        with open(global_embedding_file, 'r') as file:
            global_embeding_dict = json.load(file)
        logger.info(f'加载全局embedding文件，包含{len(global_embeding_dict)}条记录')
    else:
        global_embeding_dict = {}
        logger.info('全局embedding文件不存在')
    
    # 检查是否存在当前目录的embedding字典（保存到脚本所在目录）
    embedding_output_file = os.path.join(script_dir, 'embeding_QF_NIA.json')
    if os.path.exists(embedding_output_file):
        with open(embedding_output_file, 'r') as file:
            embeding_dict = json.load(file)
        logger.info(f'加载本地embedding文件，包含{len(embeding_dict)}条记录')
    else:
        embeding_dict = {}
        logger.info('创建新的embedding字典')
    
    logger.info(f'Embedding输出文件: {embedding_output_file}')
    
    # 设置输出路径
    output_path = '/home/lz/PycharmProjects/Pearl/test_rl/test_QF_NIA/cvc5_process_QF_NIA/features'
    logic_name = 'QF_NIA'
    output_path_file = os.path.join(output_path, logic_name + '_llm_embeddings')
    
    if not os.path.exists(output_path_file):
        os.makedirs(output_path_file)
        logger.info(f'创建输出目录: {output_path_file}')
    
    # 统计信息
    total_files = len(result_dict)
    processed_files = 0
    skipped_files = 0
    error_files = 0
    reused_embeddings = 0
    
    # 遍历所有QF_NIA文件
    for (i, (file_path, v)) in enumerate(tqdm.tqdm(result_dict.items())):
        logger.info(f'处理进度: {i+1}/{total_files} - {file_path}')
        
        # 检查文件路径是否为QF_NIA类型
        if 'QF_NIA' not in file_path:
            logger.info(f'跳过非QF_NIA文件: {file_path}')
            skipped_files += 1
            continue
        
        # 生成npy文件路径
        npy_filename = file_path.replace("/", "_") + '.npy'
        npy_path = os.path.join(output_path_file, npy_filename)
        
        # 检查是否可以从全局embedding字典中复用
        if file_path in global_embeding_dict:
            global_entry = global_embeding_dict[file_path]
            # 检查是否是有效的embedding（不是解析错误）
            if isinstance(global_entry, list) and len(global_entry) >= 3 and global_entry[0] != '解析错误':
                existing_embedding_path = global_entry[0]
                # 验证embedding文件是否真实存在
                if os.path.exists(existing_embedding_path):
                    logger.info(f'从全局字典复用embedding: {file_path}')
                    
                    # 直接使用已有的embedding路径
                    embeding_dict[file_path] = [existing_embedding_path]
                    
                    # 重新计算label（基于cvc5的求解结果）
                    # 记录可解性标签 (0: sat, 1: unsat/unknown)
                    if v[0] == 'sat':
                        embeding_dict[file_path].append(0)
                    else:
                        embeding_dict[file_path].append(1)
                    
                    # 记录时间分类标签
                    time_value = float(v[1])
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
                    
                    reused_embeddings += 1
                    processed_files += 1
                    
                    # 定期保存embedding字典
                    if processed_files % 10 == 0:
                        with open(embedding_output_file, 'w', encoding='utf-8') as file:
                            json.dump(embeding_dict, file, ensure_ascii=False, indent=4)
                        logger.info(f'保存进度: 已处理{processed_files}个文件（复用{reused_embeddings}个）')
                    
                    continue
                else:
                    logger.warning(f'全局embedding文件不存在: {existing_embedding_path}，将重新生成')
            else:
                logger.info(f'全局embedding无效（解析错误），将重新生成: {file_path}')
        
        # 判断是否已经处理过
        if os.path.exists(npy_path):
            logger.info(f'文件已处理，跳过: {file_path}')
            skipped_files += 1
            continue
        
        # 读取SMT文件内容
        try:
            with open(file_path, 'r') as file:
                smtlib_str = file.read()
        except Exception as e:
            logger.error(f'读取文件失败: {file_path}, 错误: {e}')
            error_files += 1
            continue
        
        # 归一化SMT文件
        try:
            logger.info(f'开始归一化约束文件: {time.time()}')
            temp = time.time()
            
            # 设置超时信号
            signal.alarm(30)
            signal.signal(signal.SIGALRM, timeout_handler)
            
            try:
                smtlib_str, var_dict, _ = normalize_smt_str(smtlib_str)
                logger.info(f'归一化成功，耗时: {time.time()-temp}秒')
            except MyException as e:
                logger.info(f'归一化超时: {file_path}')
                error_files += 1
                continue
            except Exception as e:
                logger.error(f'归一化失败: {file_path}, 错误: {e}')
                error_files += 1
                continue
            finally:
                signal.alarm(0)
                
        except Exception as e:
            logger.error(f'归一化过程出错: {file_path}, 错误: {e}')
            embeding_dict[file_path] = ['解析错误']
            error_files += 1
            continue
        
        # 生成embedding
        try:
            temp = time.time()
            logger.info(f'开始生成embedding: {temp}')
            embeddings = process_embeding(smtlib_str, llm_host=llm_host, llm_model=llm_model)
            logger.info(f'Embedding生成成功，耗时: {time.time()-temp}秒')
            
            # 转换为numpy数组
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.detach().numpy()
            
            # 保存embedding
            np.save(npy_path, embeddings)
            logger.info(f'Embedding已保存: {npy_path}')
            
            # 记录到embedding字典
            embeding_dict[file_path] = [npy_path]
            
            # 记录可解性标签 (0: sat, 1: unsat/unknown)
            if v[0] == 'sat':
                embeding_dict[file_path].append(0)
            else:
                embeding_dict[file_path].append(1)
            
            # 记录时间分类标签
            time_value = float(v[1])
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
            
            processed_files += 1
            
            # 定期保存embedding字典
            if processed_files % 10 == 0:
                with open(embedding_output_file, 'w', encoding='utf-8') as file:
                    json.dump(embeding_dict, file, ensure_ascii=False, indent=4)
                logger.info(f'保存进度: 已处理{processed_files}个文件')
                
        except Exception as e:
            logger.error(f'Embedding生成失败: {file_path}, 错误: {e}')
            error_files += 1
            continue
    
    # 最终保存
    with open(embedding_output_file, 'w', encoding='utf-8') as file:
        json.dump(embeding_dict, file, ensure_ascii=False, indent=4)
    
    # 输出统计信息
    logger.info('='*50)
    logger.info(f'处理完成！')
    logger.info(f'总文件数: {total_files}')
    logger.info(f'成功处理: {processed_files}')
    logger.info(f'  - 复用已有embedding: {reused_embeddings}')
    logger.info(f'  - 新生成embedding: {processed_files - reused_embeddings}')
    logger.info(f'跳过文件: {skipped_files}')
    logger.info(f'错误文件: {error_files}')
    logger.info(f'Embedding字典已保存到: {embedding_output_file}')
    logger.info('='*50)

def split_train_test_QF_NIA(embedding_dict_path=None):
    """
    基于已有的训练/测试集划分，使用embeding_QF_NIA.json中已计算的label并处理新增文件
    
    Args:
        embedding_dict_path: embedding字典文件路径（默认为脚本所在目录的embeding_QF_NIA.json）
    """
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 如果未指定路径，使用默认路径
    if embedding_dict_path is None:
        embedding_dict_path = os.path.join(script_dir, 'embeding_QF_NIA.json')
    
    # 读取当前embedding字典（已包含最新的label）
    logger.info(f'读取embedding字典: {embedding_dict_path}')
    with open(embedding_dict_path, 'r') as file:
        embeding_dict = json.load(file)
    
    # 读取原始的训练/测试集划分
    original_train_path = '/home/lz/PycharmProjects/Pearl/test_rl/predictor/smt_comp_NIA/QF_NIA_train.json'
    original_test_path = '/home/lz/PycharmProjects/Pearl/test_rl/predictor/smt_comp_NIA/QF_NIA_test.json'
    
    logger.info(f'读取原始训练集: {original_train_path}')
    with open(original_train_path, 'r') as file:
        original_train = json.load(file)
    
    logger.info(f'读取原始测试集: {original_test_path}')
    with open(original_test_path, 'r') as file:
        original_test = json.load(file)
    
    # 过滤掉错误的embedding条目
    valid_embedding_keys = set(k for k, v in embeding_dict.items() 
                               if isinstance(v, list) and len(v) >= 3 and v[0] != '解析错误')
    
    logger.info(f'有效embedding数量: {len(valid_embedding_keys)}')
    
    # 获取原始划分的键集合
    original_train_keys = set(original_train.keys())
    original_test_keys = set(original_test.keys())
    all_original_keys = original_train_keys | original_test_keys
    
    logger.info(f'原始训练集大小: {len(original_train_keys)}')
    logger.info(f'原始测试集大小: {len(original_test_keys)}')
    
    # 分析差异
    # 1. 原始划分中存在但embedding中不存在的键（需要删除）
    keys_to_remove = all_original_keys - valid_embedding_keys
    
    # 2. embedding中存在但原始划分中不存在的键（需要添加）
    keys_to_add = valid_embedding_keys - all_original_keys
    
    logger.info(f'需要删除的键（原始有但embedding没有）: {len(keys_to_remove)}')
    logger.info(f'需要添加的键（embedding有但原始没有）: {len(keys_to_add)}')
    
    # 构建新的训练集和测试集
    new_train = {}
    new_test = {}
    
    # 1. 处理原始训练集中的键（直接使用embeding_dict中的数据）
    for key in original_train_keys:
        if key in valid_embedding_keys:
            # 直接使用embeding_dict中已经计算好的值 [embedding_path, solvability_label, time_label]
            new_train[key] = embeding_dict[key]
    
    # 2. 处理原始测试集中的键（直接使用embeding_dict中的数据）
    for key in original_test_keys:
        if key in valid_embedding_keys:
            # 直接使用embeding_dict中已经计算好的值
            new_test[key] = embeding_dict[key]
    
    # 3. 处理新增的键（按1:1比例分配）
    new_train_added = 0
    new_test_added = 0
    if keys_to_add:
        logger.info(f'按1:1比例分配{len(keys_to_add)}个新键到训练集和测试集')
        new_keys_list = list(keys_to_add)
        random.shuffle(new_keys_list)
        
        mid_point = len(new_keys_list) // 2
        new_train_keys = new_keys_list[:mid_point]
        new_test_keys = new_keys_list[mid_point:]
        
        for key in new_train_keys:
            # 直接使用embeding_dict中已经计算好的值
            new_train[key] = embeding_dict[key]
            new_train_added += 1
        
        for key in new_test_keys:
            # 直接使用embeding_dict中已经计算好的值
            new_test[key] = embeding_dict[key]
            new_test_added += 1
    
    # 保存到脚本所在目录
    train_path = os.path.join(script_dir, 'QF_NIA_train.json')
    test_path = os.path.join(script_dir, 'QF_NIA_test.json')
    
    with open(train_path, 'w', encoding='utf-8') as file:
        json.dump(new_train, file, ensure_ascii=False, indent=4)
    
    with open(test_path, 'w', encoding='utf-8') as file:
        json.dump(new_test, file, ensure_ascii=False, indent=4)
    
    # 输出统计信息
    logger.info('='*70)
    logger.info(f'数据集更新完成！')
    logger.info(f'删除的键数量: {len(keys_to_remove)}')
    logger.info(f'新增的键数量: {len(keys_to_add)} (训练集: {new_train_added}, 测试集: {new_test_added})')
    logger.info(f'训练集: {len(new_train)} 个样本 -> {train_path}')
    logger.info(f'测试集: {len(new_test)} 个样本 -> {test_path}')
    logger.info(f'总计: {len(new_train) + len(new_test)} 个样本')
    logger.info('='*70)
    
    return new_train, new_test

if __name__ == '__main__':
    setup_logger()
    logger.add('embedding_generation.log')
    # 生成embeddings
    test_group_bert_normalize_1by1_smt_name_2_QF_NIA()
    
    # 分割训练集和测试集
    split_train_test_QF_NIA()

