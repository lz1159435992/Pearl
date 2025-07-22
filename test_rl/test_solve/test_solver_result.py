import json
import os

import time

from overrides.typing_utils import unknown
from z3 import *
from z3.z3 import parse_smt2_string, Solver

from test_rl.test_script.utils import solve_and_measure_time, model_to_dict, load_dictionary, setup_logger, \
    normalize_smt_str
from loguru import logger

# 添加SuperVenn绘制所需的导入
import matplotlib.pyplot as plt
from supervenn import supervenn


os.environ['ALL_PROXY'] = ''
os.environ['all_proxy'] = ''
setup_logger()

def plot_supervenn_diagrams_cvc5(baseline_solved, rl_llm_solved, solver_name="CVC5", output_dir=None):
    """
    绘制SuperVenn图，比较baseline求解器和RL+LLM增强版本的求解能力

    Args:
        baseline_solved: baseline求解器解决的约束集合
        rl_llm_solved: RL+LLM增强版本解决的约束集合
        solver_name: 求解器名称，用于标签
        output_dir: 输出目录，如果为None则不保存文件
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    set_labels = [f'{solver_name} Baseline', f'{solver_name} + RL+LLM']

    def plot_supervenn(sets_data, title, filename):
        # Professional color palette
        color_palette = ['#0072B2', '#D55E00']

        try:
            with plt.style.context('seaborn-v0_8-paper'):
                plt.rcParams.update({
                    'font.family': 'serif',
                    'font.serif': ['Times New Roman', 'DejaVu Serif'],
                    'font.size': 10
                })
                plt.figure(figsize=(10, 3))
                plot = supervenn(sets_data,
                          set_annotations=set_labels,
                          side_plots=True,
                          sets_ordering='minimize gaps',
                          bar_height=1.0,
                          side_plot_width=0.4,
                          color_cycle=color_palette,
                          rotate_col_annotations=False,
                          widths_minmax_ratio=0.02)

                if 'main' in plot.axes:
                    plot.axes['main'].set_title(title, fontsize=14, fontweight='bold')

                if output_dir:
                    output_path = os.path.join(output_dir, filename)
                    plt.savefig(output_path, bbox_inches='tight', format='pdf', dpi=300)
                    print(f"SuperVenn diagram saved to: {output_path}")
                else:
                    plt.show()
                plt.close()
        except Exception as e:
            print(f"Error creating SuperVenn diagram: {e}")
            plt.close()

    # 创建求解能力比较图
    solved_sets = [baseline_solved, rl_llm_solved]
    plot_supervenn(solved_sets, f"{solver_name} Solving Capability Comparison", f"supervenn_{solver_name.lower()}_comparison.pdf")

    # 打印统计信息
    baseline_only = baseline_solved - rl_llm_solved
    rl_llm_only = rl_llm_solved - baseline_solved
    both_solved = baseline_solved & rl_llm_solved

    print(f"\n{solver_name} SuperVenn Analysis:")
    print(f"  {solver_name} Baseline only solved: {len(baseline_only)} constraints")
    print(f"  {solver_name} + RL+LLM only solved: {len(rl_llm_only)} constraints")
    print(f"  Both methods solved: {len(both_solved)} constraints")
    print(f"  Total unique constraints solved: {len(baseline_solved | rl_llm_solved)} constraints")

    return {
        'baseline_only': baseline_only,
        'rl_llm_only': rl_llm_only,
        'both_solved': both_solved,
        'baseline_total': len(baseline_solved),
        'rl_llm_total': len(rl_llm_solved),
        'union_total': len(baseline_solved | rl_llm_solved)
    }

def test_group():
    info_name = 'info_dict_gai_6_normal_1109_pre_SMTimer_llama3.1:70b_1200s_result.txt'
    if not os.path.exists(info_name):
        # 文件不存在时，创建文件
        info_dict = {}
        with open(info_name, 'w') as file:
            json.dump(info_dict, file, indent=4)
        print(f'文件{info_name} 已创建。')
    else:
        info_dict = load_dictionary(info_name)
        print(f'文件已存在。')
    with open('/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1109_pre_SMTimer_llama3.1:70b_1200s.txt', 'r') as file:
        result_dict = json.load(file)

    for key, value in result_dict.items():
        if '/home/yy/Downloads/' in key:
            file_path = key.replace('/home/yy/Downloads/', '/home/lz/baidudisk/')
        elif '/home/nju/Downloads/' in key:
            file_path = key.replace('/home/nju/Downloads/', '/home/lz/baidudisk/')
        else:
            file_path = key
        with open(file_path, 'r') as file:
            # 读取文件所有内容到一个字符串
            smtlib_str = file.read()

        dict_obj = json.loads(smtlib_str)
        if 'smt-comp' in file_path:
            smtlib_str = dict_obj['smt_script']
        else:
            smtlib_str = dict_obj['script']
            # python_list = json.loads(value)
        if file_path not in info_dict.keys():
            assertions = parse_smt2_string(smtlib_str)
            solver = Solver()
            for a in assertions:
                solver.add(a)
            # timeout = 999999999
            timeout = 1200000

            #先取消求解
            result, model, time_taken = solve_and_measure_time(solver, timeout)
            print(result_dict[key][1])
            result_dict[key][1] = time_taken
            print(result, time_taken)
            # if result == 'unsat' or time_taken < 500:
            #     continue
            # result_list = [result, time_taken, timeout]
            #
            # if model:
            #     result_list.append(model_to_dict(model))
            # print(result_list[-1])
            #
            # info_dict[file_path] = result_list
            info_dict[key] = result_dict[key]
            with open(info_name, 'w') as file:
                json.dump(info_dict, file, indent=4)
        # elif file_path not in info_dict.keys() and python_list[0] == 'unsat':
        #     info_dict[file_path] = python_list
        #     with open(info_name, 'w') as file:
        #         json.dump(info_dict, file, indent=4)
#选择修正过的求解时间大于300s的测试数据
def test_group_1():
    info_name = 'info_dict_gai_6_normal_1109_pre_SMTimer_llama3.1:70b_1200s_result.txt'
    if not os.path.exists(info_name):
        # 文件不存在时，创建文件
        info_dict = {}
        with open(info_name, 'w') as file:
            json.dump(info_dict, file, indent=4)
        print(f'文件{info_name} 已创建。')
    else:
        info_dict = load_dictionary(info_name)
        print(f'文件已存在。')
    count = 0
    succeed_count = 0
    failed_count = 0
    all_time_sat_succeed = 0
    all_time_rl_succeed = 0
    all_time_sat_failed = 0
    all_time_rl_failed = 0
    all_time_sat = 0
    all_time_rl = 0
    for k,v in info_dict.items():
       if v[1] > 300:
            print(k,v[0],v[1],v[2],v[3],v[4])
            count += 1
            all_time_sat += v[1]
            all_time_rl += v[3]
            if v[4] == 'succeed':
                succeed_count += 1
                all_time_sat_succeed += v[1]
                all_time_rl_succeed += v[3]
            elif v[4] == 'failed':
                failed_count += 1
                all_time_sat_failed += v[1]
                all_time_rl_failed += v[3]
    print(count,succeed_count,failed_count,all_time_sat,all_time_rl,all_time_sat_succeed,all_time_rl_succeed,all_time_sat_failed,all_time_rl_failed)

#选择修正过的求解时间大于300s的测试数据
def test_group_1_save():
    solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict.txt'
    solve_dict = load_dictionary(solve_name)

    info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1109_pre_SMTimer_save_docker_llama_3.1:70b_1200s.txt'
    info_dict = load_dictionary(info_name)
    for k,v in info_dict.items():
        if k in solve_dict.keys():
            info_dict[k][0] = solve_dict[k][0]
            info_dict[k][1] = solve_dict[k][1]

    # if not os.path.exists(info_name):
    #     # 文件不存在时，创建文件
    #     info_dict = {}
    #     with open(info_name, 'w') as file:
    #         json.dump(info_dict, file, indent=4)
    #     print(f'文件{info_name} 已创建。')
    # else:
    #     info_dict = load_dictionary(info_name)
    #     print(f'文件已存在。')
    count = 0
    succeed_count = 0
    failed_count = 0
    all_time_sat_succeed = 0
    all_time_rl_succeed = 0
    all_time_sat_failed = 0
    all_time_rl_failed = 0
    all_time_sat = 0
    all_time_rl = 0
    for k,v in info_dict.items():
       if v[1] > 300:
            print(k,v[0],v[1],v[2],v[3],v[4])
            count += 1
            all_time_sat += v[1]
            all_time_rl += v[4]
            if v[5] == 'succeed':
                succeed_count += 1
                all_time_sat_succeed += v[1]
                all_time_rl_succeed += v[4]
            elif v[5] == 'failed':
                failed_count += 1
                all_time_sat_failed += v[1]
                all_time_rl_failed += v[4]
    print(count,succeed_count,failed_count,all_time_sat,all_time_rl,all_time_sat_succeed,all_time_rl_succeed,all_time_sat_failed,all_time_rl_failed)

#选择修正过的求解时间大于300s的测试数据
def test_group_2_save_1208(solve_name,info_name):

    solve_dict = load_dictionary(solve_name)

    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1109_pre_SMTimer_save_docker_llama_3.1:70b_1200s.txt'

    info_dict = load_dictionary(info_name)
    for k,v in info_dict.items():
        if k in solve_dict.keys():
            info_dict[k][0] = solve_dict[k][0]
            info_dict[k][1] = solve_dict[k][1]
            info_dict[k][2] = solve_dict[k][2]
    # if not os.path.exists(info_name):
    #     # 文件不存在时，创建文件
    #     info_dict = {}
    #     with open(info_name, 'w') as file:
    #         json.dump(info_dict, file, indent=4)
    #     print(f'文件{info_name} 已创建。')
    # else:
    #     info_dict = load_dictionary(info_name)
    #     print(f'文件已存在。')
    count = 0
    succeed_count = 0
    failed_count = 0
    all_time_sat_succeed = 0
    all_time_rl_succeed = 0
    all_time_sat_failed = 0
    all_time_rl_failed = 0
    all_time_sat = 0
    all_time_rl = 0
    unknown_count = 0
    sat_count = 0
    unknown2succeed = 0
    unknown2failed = 0
    sat2succeed = 0
    sat2failed = 0

    # v0 原始求解是否成功
    # v1 实际求解时间
    # v2 求解超时时间
    # v3 大模型求解时间
    # v4 大模型求解是否成功
    time_out = 500
    for k, v in info_dict.items():
        if v[1] > time_out:
            v[0] = 'unknown'
            v[1] = time_out
        if v[4] > time_out:
            v[5] = 'failed'
            v[4] = time_out

    for k,v in info_dict.items():

        if v[1] > 300:
            if v[0] == 'sat' and v[1] <= time_out:
                sat_count += 1
                if v[5] == 'succeed':
                    sat2succeed += 1
                else:
                    sat2failed += 1
            elif v[0] == 'unknown':
            # elif v[1]>time_out:
                print(k,v)
                unknown_count += 1
                if v[5] == 'succeed':
                    unknown2succeed += 1
                else:
                    unknown2failed += 1
            # print(k,v[0],v[1],v[2],v[3],v[4])
            count += 1
            all_time_sat += v[1]
            all_time_rl += v[4]
            if v[5] == 'succeed':
                succeed_count += 1
                all_time_sat_succeed += v[1]
                all_time_rl_succeed += v[4]
            elif v[5] == 'failed':
                failed_count += 1
                all_time_sat_failed += v[1]
                all_time_rl_failed += v[4]
    print(f"Total Count: {count},\n Succeed Count: {succeed_count},\n Failed Count: {failed_count},\n "
          f"All Time SAT: {all_time_sat},\n All Time RL: {all_time_rl},\n "
          f"All Time SAT Succeed: {all_time_sat_succeed},\n All Time RL Succeed: {all_time_rl_succeed},\n "
          f"All Time SAT Failed: {all_time_sat_failed},\n All Time RL Failed: {all_time_rl_failed}")
    print(f"Unknown Count: {unknown_count},\n SAT Count: {sat_count},\n "
          f"Unknown to Succeed: {unknown2succeed},\n Unknown to Failed: {unknown2failed},\n "
          f"SAT to Succeed: {sat2succeed},\n SAT to Failed: {sat2failed}\n")
    #修改求解时间,求解时间大于1200s的测试数据记为unknown
#选择修正过的求解时间大于300s的测试数据
def test_group_1_no_save():
    solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_bingxing.txt'
    solve_dict = load_dictionary(solve_name)

    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1109_pre_SMTimer_llama3.1:70b_1200s.txt'
    info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1110_pre_SMTimer_llama3.1:70b_1200s_info_dict_rl.txt'

    info_dict = load_dictionary(info_name)
    for k,v in info_dict.items():
        if k in solve_dict.keys():
            info_dict[k][0] = solve_dict[k][0]
            info_dict[k][1] = solve_dict[k][1]
            info_dict[k][2] = solve_dict[k][2]
    # if not os.path.exists(info_name):
    #     # 文件不存在时，创建文件
    #     info_dict = {}
    #     with open(info_name, 'w') as file:
    #         json.dump(info_dict, file, indent=4)
    #     print(f'文件{info_name} 已创建。')
    # else:
    #     info_dict = load_dictionary(info_name)
    #     print(f'文件已存在。')
    count = 0
    succeed_count = 0
    failed_count = 0
    all_time_sat_succeed = 0
    all_time_rl_succeed = 0
    all_time_sat_failed = 0
    all_time_rl_failed = 0
    all_time_sat = 0
    all_time_rl = 0
    unknown_count = 0
    sat_count = 0
    unknown2succeed = 0
    unknown2failed = 0
    sat2succeed = 0
    sat2failed = 0
    for k,v in info_dict.items():

        if v[1] > 300:
            if v[0] == 'sat' and v[1] <= 1200:
                sat_count += 1
                if v[4] == 'succeed':
                    sat2succeed += 1
                else:
                    sat2failed += 1
            elif v[1]>1200:
                print(k,v)
                unknown_count += 1
                if v[4] == 'succeed':
                    unknown2succeed += 1
                else:
                    unknown2failed += 1
            # print(k,v[0],v[1],v[2],v[3],v[4])
            count += 1
            all_time_sat += v[1]
            all_time_rl += v[3]
            if v[4] == 'succeed':
                succeed_count += 1
                all_time_sat_succeed += v[1]
                all_time_rl_succeed += v[3]
            elif v[4] == 'failed':
                failed_count += 1
                all_time_sat_failed += v[1]
                all_time_rl_failed += v[3]
    print(f"Total Count: {count}, Succeed Count: {succeed_count}, Failed Count: {failed_count}, "
          f"All Time SAT: {all_time_sat}, All Time RL: {all_time_rl}, "
          f"All Time SAT Succeed: {all_time_sat_succeed}, All Time RL Succeed: {all_time_rl_succeed}, "
          f"All Time SAT Failed: {all_time_sat_failed}, All Time RL Failed: {all_time_rl_failed}")
    print(f"Unknown Count: {unknown_count}, SAT Count: {sat_count}, "
          f"Unknown to Succeed: {unknown2succeed}, Unknown to Failed: {unknown2failed}, "
          f"SAT to Succeed: {sat2succeed}, SAT to Failed: {sat2failed}")
    #修改求解时间,求解时间大于1200s的测试数据记为unknown
#选择修正过的求解时间大于300s的测试数据
def test_group_1_llm():
    solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict.txt'
    solve_dict = load_dictionary(solve_name)

    info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_normal_1109_llm_no_rl_direct_solve_docker_llama3.1:70b_1set_1200s.txt'
    info_dict = load_dictionary(info_name)
    for k,v in info_dict.items():
        if k in solve_dict.keys():
            info_dict[k][0] = solve_dict[k][0]
            info_dict[k][1] = solve_dict[k][1]
            # info_dict[k][2] = solve_dict[k][2]
    # if not os.path.exists(info_name):
    #     # 文件不存在时，创建文件
    #     info_dict = {}
    #     with open(info_name, 'w') as file:
    #         json.dump(info_dict, file, indent=4)
    #     print(f'文件{info_name} 已创建。')
    # else:
    #     info_dict = load_dictionary(info_name)
    #     print(f'文件已存在。')
    count = 0
    succeed_count = 0
    failed_count = 0
    all_time_sat_succeed = 0
    all_time_rl_succeed = 0
    all_time_sat_failed = 0
    all_time_rl_failed = 0
    all_time_sat = 0
    all_time_rl = 0
    unknown_count = 0
    sat_count = 0
    unknown2succeed = 0
    unknown2failed = 0
    sat2succeed = 0
    sat2failed = 0
    for k,v in info_dict.items():

        if v[1] > 300:
            if v[0] == 'sat' and v[1] <= 1200:
                sat_count += 1
                if v[3] == 'succeed':
                    sat2succeed += 1
                else:
                    sat2failed += 1
            elif v[1]>1200:
                print(k,v)
                unknown_count += 1
                if v[3] == 'succeed':
                    unknown2succeed += 1
                else:
                    unknown2failed += 1
            # print(k,v[0],v[1],v[2],v[3],v[4])
            count += 1
            all_time_sat += v[1]

            if v[3] == 'succeed':
                succeed_count += 1
                all_time_sat_succeed += v[1]
                all_time_rl_succeed += v[6]
                all_time_rl += v[6]
            elif v[3] == 'failed':
                failed_count += 1
                all_time_sat_failed += v[1]
                all_time_rl_failed += v[4]
                all_time_rl += v[4]
    print(f"Total Count: {count}, Succeed Count: {succeed_count}, Failed Count: {failed_count}, "
          f"All Time SAT: {all_time_sat}, All Time RL: {all_time_rl}, "
          f"All Time SAT Succeed: {all_time_sat_succeed}, All Time RL Succeed: {all_time_rl_succeed}, "
          f"All Time SAT Failed: {all_time_sat_failed}, All Time RL Failed: {all_time_rl_failed}")
    print(f"Unknown Count: {unknown_count}, SAT Count: {sat_count}, "
          f"Unknown to Succeed: {unknown2succeed}, Unknown to Failed: {unknown2failed}, "
          f"SAT to Succeed: {sat2succeed}, SAT to Failed: {sat2failed}")

    print(len(info_dict),len(solve_dict))
    #修改求解时间,求解时间大于1200s的测试数据记为unknown
def test_group_2():
    info_name = 'info_dict_gai_6_normal_1109_pre_SMTimer_llama3.1:70b_1200s_result.txt'
    if not os.path.exists(info_name):
        # 文件不存在时，创建文件
        info_dict = {}
        with open(info_name, 'w') as file:
            json.dump(info_dict, file, indent=4)
        print(f'文件{info_name} 已创建。')
    else:
        info_dict = load_dictionary(info_name)
        print(f'文件已存在。')
    count = 0
    succeed_count = 0
    failed_count = 0
    all_time_sat = 0
    all_time_rl = 0
    for k,v in info_dict.items():
       if v[1] > 1200:
            # info_dict[k][0] = 'unknown'
            print(k,v[0],v[1],v[2],v[3],v[4])
            count += 1
    print(count)
    # with open(info_name, 'w') as file:
    #     json.dump(info_dict, file, indent=4)
    # 修改求解时间,求解时间大于1200s的测试数据记为unknown
def test_group_3():
    with open('/home/lz/sibyl_3/src/networks/info_dict_rl.txt', 'r') as file:
        rl_dict = json.load(file)
    info_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict.txt'
    if not os.path.exists(info_name):
        # 文件不存在时，创建文件
        info_dict = {}
        with open(info_name, 'w') as file:
            json.dump(info_dict, file, indent=4)
        print(f'文件{info_name} 已创建。')
    else:
        info_dict = load_dictionary(info_name)
        print(f'文件已存在。')
    count = 0
    succeed_count = 0
    failed_count = 0
    all_time_sat = 0
    all_time_rl = 0
    for k, v in info_dict.items():
        if (v[0]=='sat' or v[0]=='unknown') and v[1] > 300 and k in rl_dict.keys():
            # info_dict[k][0] = 'unknown'
            # print(k, v[0], v[1], v[2], v[3], v[4])
            print(k,v)
            count += 1
    print(count)
    # with open(info_name, 'w') as file:
    #     json.dump(info_dict, file, indent=4)
def extract_unique_keys(file_paths):
    unique_keys = set()  # 使用集合来存储不重复的键

    for file_path in file_paths:
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)  # 加载JSON数据
                unique_keys.update(data.keys())  # 更新集合，添加新的键
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {file_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    return unique_keys
def something():
    solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict.txt'
    solve_dict = load_dictionary(solve_name)

    info_name = '/home/lz/sibyl_3/src/networks/info_dict_rl.txt'
    info_dict = load_dictionary(info_name)

    count = 0
    for k,v in info_dict.items():
        if (info_dict[k][0] == 'sat' or info_dict[k][0] == 'unknown') and info_dict[k][1] > 300:
            count += 1
        # if k in solve_dict.keys():
        #     if (solve_dict[k][0] == 'sat' or solve_dict[k][0] == 'unknown') and solve_dict[k][1] > 300:
        #         count += 1
    print(len(solve_dict),len(info_dict),count)
    # 定义文件路径列表
    file_paths = [
        '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_normal_1103_llm_no_rl_direct_solve_docker_llama3.1:70b_1set.txt',
        '/home/lz/sibyl_3/src/networks/info_dict_rl.txt',
        '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_109_2_SMTimer.txt',
        # '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict.txt'
    ]

    # 调用函数并打印结果
    unique_keys = extract_unique_keys(file_paths)
    print("Unique keys across all files:")
    print(len(unique_keys))
    count = 0
    for k in unique_keys:
        if k in solve_dict.keys():
            count += 1
    print(count)
    # for key in sorted(unique_keys):  # 排序输出
    #     print(key)

    solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_normal_1109_llm_no_rl_direct_solve_docker_llama3.1:70b_1set_1200s.txt'
    solve_dict = load_dictionary(solve_name)

    info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1110_pre_SMTimer_llama3.1:70b_1200s_info_dict_rl.txt'
    info_dict = load_dictionary(info_name)
    # 提取字典的键
    solve_keys = set(solve_dict.keys())
    info_keys = set(info_dict.keys())

    # 计算交集和并集
    keys_intersection = solve_keys.intersection(info_keys)
    keys_union = solve_keys.union(info_keys)

    print(len(solve_dict),len(info_dict))
    # 打印结果
    print("Keys Intersection:", len(keys_intersection))

    print("Keys Union:", len(keys_union))
    print(keys_union)

    info_dict = load_dictionary('/home/lz/PycharmProjects/Pearl/test_rl/result_dict.txt')
    print(len(info_dict))
def test_group_2_no_save(solve_name,info_name):

    solve_dict = load_dictionary(solve_name)

    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1109_pre_SMTimer_llama3.1:70b_1200s.txt'

    info_dict = load_dictionary(info_name)
    for k,v in info_dict.items():
        if k in solve_dict.keys():
            info_dict[k][0] = solve_dict[k][0]
            info_dict[k][1] = solve_dict[k][1]
            info_dict[k][2] = solve_dict[k][2]
    # if not os.path.exists(info_name):
    #     # 文件不存在时，创建文件
    #     info_dict = {}
    #     with open(info_name, 'w') as file:
    #         json.dump(info_dict, file, indent=4)
    #     print(f'文件{info_name} 已创建。')
    # else:
    #     info_dict = load_dictionary(info_name)
    #     print(f'文件已存在。')
    result_dict = {}
    result_dict['succeed'] = []
    result_dict['failed'] = []
    result_dict['unknown to succeed'] = []
    result_dict['unknown to failed'] = []
    result_dict['sat to succeed'] = []
    result_dict['sat to failed'] = []
    count = 0
    succeed_count = 0
    failed_count = 0
    all_time_sat_succeed = 0
    all_time_rl_succeed = 0
    all_time_sat_failed = 0
    all_time_rl_failed = 0
    all_time_sat = 0
    all_time_rl = 0
    unknown_count = 0
    sat_count = 0
    unknown2succeed = 0
    unknown2failed = 0
    sat2succeed = 0
    sat2failed = 0
    for k,v in info_dict.items():

        if v[1] > 300:
            if v[0] == 'sat' and v[1] <= 1200:
                sat_count += 1
                if v[4] == 'succeed':
                    result_dict['succeed'].append(k)
                    sat2succeed += 1
                    result_dict['sat to succeed'].append(k)
                else:
                    result_dict['failed'].append(k)
                    sat2failed += 1
                    result_dict['sat to failed'].append(k)
            elif v[1]>1200:
                # print(k,v)
                unknown_count += 1
                if v[4] == 'succeed':
                    result_dict['succeed'].append(k)
                    unknown2succeed += 1
                    result_dict['unknown to succeed'].append(k)
                else:
                    result_dict['failed'].append(k)
                    unknown2failed += 1
                    result_dict['unknown to failed'].append(k)
            # print(k,v[0],v[1],v[2],v[3],v[4])
            count += 1
            all_time_sat += v[1]
            all_time_rl += v[3]
            if v[4] == 'succeed':
                succeed_count += 1
                all_time_sat_succeed += v[1]
                all_time_rl_succeed += v[3]
            elif v[4] == 'failed':
                failed_count += 1
                all_time_sat_failed += v[1]
                all_time_rl_failed += v[3]
    # logger.info(result_dict)
    print(f"Total Count: {count}, Succeed Count: {succeed_count}, Failed Count: {failed_count}, "
          f"All Time SAT: {all_time_sat}, All Time RL: {all_time_rl}, "
          f"All Time SAT Succeed: {all_time_sat_succeed}, All Time RL Succeed: {all_time_rl_succeed}, "
          f"All Time SAT Failed: {all_time_sat_failed}, All Time RL Failed: {all_time_rl_failed}")
    print(f"Unknown Count: {unknown_count}, SAT Count: {sat_count}, "
          f"Unknown to Succeed: {unknown2succeed}, Unknown to Failed: {unknown2failed}, "
          f"SAT to Succeed: {sat2succeed}, SAT to Failed: {sat2failed}")
    print(result_dict)
    return result_dict
    #修改求解时间,求解时间大于1200s的测试数据记为unknown
def test_group_2_no_save_1207(solve_name, info_name, var_count_path='/home/lz/PycharmProjects/Pearl/test_rl/test_solve/var_count.txt'):
    time_dict = {}
    time_dict_2 = {}
    solve_dict = load_dictionary(solve_name)

    var_count = load_dictionary(var_count_path)
    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1109_pre_SMTimer_llama3.1:70b_1200s.txt'

    info_dict = load_dictionary(info_name)
    for k,v in info_dict.items():
        if k in solve_dict.keys():
            info_dict[k][0] = solve_dict[k][0]
            info_dict[k][1] = solve_dict[k][1]
            info_dict[k][2] = solve_dict[k][2]
    # if not os.path.exists(info_name):
    #     # 文件不存在时，创建文件
    #     info_dict = {}
    #     with open(info_name, 'w') as file:
    #         json.dump(info_dict, file, indent=4)
    #     print(f'文件{info_name} 已创建。')
    # else:
    #     info_dict = load_dictionary(info_name)
    #     print(f'文件已存在。')
    result_dict = {}
    result_dict['succeed'] = []
    result_dict['failed'] = []
    result_dict['unknown to succeed'] = []
    result_dict['unknown to failed'] = []
    result_dict['sat to succeed'] = []
    result_dict['sat to failed'] = []
    count = 0
    succeed_count = 0
    failed_count = 0
    all_time_sat_succeed = 0
    all_time_rl_succeed = 0
    all_time_sat_failed = 0
    all_time_rl_failed = 0
    all_time_sat = 0
    all_time_rl = 0
    unknown_count = 0
    sat_count = 0
    unknown2succeed = 0
    unknown2failed = 0
    sat2succeed = 0
    sat2failed = 0
    z3_sat_time = 0
    z3_failed_time = 0
    #v0 原始求解是否成功
    #v1 实际求解时间
    #v2 求解超时时间
    #v3 大模型求解时间
    #v4 大模型求解是否成功
    time_out = 1200
    for k, v in info_dict.items():
        if v[1] > time_out:
            v[0] = 'unknown'
            v[1] = time_out
        if v[3] > time_out:
            v[4] = 'failed'
            v[3] = time_out
    del_list = []
    for k,v in info_dict.items():
        #筛选变量个数
        if v[1] > 300 and len(var_count[k]) > 5:
            #收集求解时间 z3solver
            if v[0] == 'sat':
                time_dict[k] = v[1]
                z3_sat_time += v[1]
            else:
                time_dict[k] = -v[1]
                z3_failed_time += v[1]
            #收集其他求解时间
            if v[4] == 'succeed':
                time_dict_2[k] = v[3]
            else:
                time_dict_2[k] = -v[3]
            if v[0] == 'sat' and v[1] <= time_out:
                sat_count += 1
                if v[4] == 'succeed':
                    result_dict['succeed'].append(k)
                    sat2succeed += 1
                    result_dict['sat to succeed'].append(k)
                else:
                    result_dict['failed'].append(k)
                    sat2failed += 1
                    result_dict['sat to failed'].append(k)
            # elif v[1]>900:
            elif v[0] == 'unknown':
                # print(k,v)
                unknown_count += 1
                if v[4] == 'succeed':
                    result_dict['succeed'].append(k)
                    unknown2succeed += 1
                    result_dict['unknown to succeed'].append(k)
                else:
                    result_dict['failed'].append(k)
                    unknown2failed += 1
                    result_dict['unknown to failed'].append(k)
            # print(k,v[0],v[1],v[2],v[3],v[4])
            count += 1
            all_time_sat += v[1]
            all_time_rl += v[3]
            if v[4] == 'succeed':
                succeed_count += 1
                all_time_sat_succeed += v[1]
                all_time_rl_succeed += v[3]
            elif v[4] == 'failed':
                failed_count += 1
                all_time_sat_failed += v[1]
                all_time_rl_failed += v[3]
        else:
            del_list.append(k)                      #删除变量个数小于5的测试数据
    for k in del_list:
        del info_dict[k]
    # logger.info(result_dict)
    print(f"Total Count: {count}, Succeed Count: {succeed_count}, Failed Count: {failed_count}, "
          f"All Time SAT: {all_time_sat},All Time SAT avg: {all_time_sat/count}  All Time RL: {all_time_rl},All Time RL avg:{all_time_rl/count}, "
          f"All Time SAT Succeed: {all_time_sat_succeed}, All Time SAT Succeed avg: {all_time_sat_succeed/succeed_count}, All Time RL Succeed: {all_time_rl_succeed},All Time RL Succeed avg: {all_time_rl_succeed/succeed_count}, "
          f"All Time SAT Failed: {all_time_sat_failed}, All Time SAT Failed avg: {all_time_sat_failed/failed_count},All Time RL Failed: {all_time_rl_failed},All Time RL Failed avg: {all_time_rl_failed/failed_count}")
    print(f"Unknown Count: {unknown_count}, SAT Count: {sat_count}, "
          f"Unknown to Succeed: {unknown2succeed}, Unknown to Failed: {unknown2failed}, "
          f"SAT to Succeed: {sat2succeed}, SAT to Failed: {sat2failed}")
    print(result_dict)
    print(time_dict)
    return result_dict,time_dict,time_dict_2,info_dict
    #修改求解时间,求解时间大于1200s的测试数据记为unknown
def test_group_2_no_save_0607_cvc5(solve_name,info_name,new_solver):
    new_solver_dict = load_dictionary(new_solver)
    for k, v in new_solver_dict.items():
        if v[0] == 'sat' and v[1] > 1200:
            new_solver_dict[k][0] = 'unknown'
            new_solver_dict[k][1] = 1200
        if (v[0] == 'unknown' or v[0] == 'timeout') and v[1] >= 1200:
            new_solver_dict[k][0] = 'unknown'
            new_solver_dict[k][1] = 1200

    time_dict = {}
    time_dict_2 = {}
    solve_dict = load_dictionary(solve_name)

    var_count = load_dictionary('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/var_count.txt')
    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1109_pre_SMTimer_llama3.1:70b_1200s.txt'

    info_dict = load_dictionary(info_name)
    for k,v in info_dict.items():
        if k in solve_dict.keys():
            info_dict[k][0] = solve_dict[k][0]
            info_dict[k][1] = solve_dict[k][1]
            info_dict[k][2] = solve_dict[k][2]
    # if not os.path.exists(info_name):
    #     # 文件不存在时，创建文件
    #     info_dict = {}
    #     with open(info_name, 'w') as file:
    #         json.dump(info_dict, file, indent=4)
    #     print(f'文件{info_name} 已创建。')
    # else:
    #     info_dict = load_dictionary(info_name)
    #     print(f'文件已存在。')
    result_dict = {}
    result_dict['succeed'] = []
    result_dict['failed'] = []
    result_dict['unknown to succeed'] = []
    result_dict['unknown to failed'] = []
    result_dict['sat to succeed'] = []
    result_dict['sat to failed'] = []
    count = 0
    succeed_count = 0
    failed_count = 0
    all_time_sat_succeed = 0
    all_time_rl_succeed = 0
    all_time_sat_failed = 0
    all_time_rl_failed = 0
    all_time_sat = 0
    all_time_rl = 0
    unknown_count = 0
    sat_count = 0
    unknown2succeed = 0
    unknown2failed = 0
    sat2succeed = 0
    sat2failed = 0
    z3_sat_time = 0
    z3_failed_time = 0

    all_time_newsolver_succeed = 0
    all_time_newsolver_failed = 0
    #v0 原始求解是否成功
    #v1 实际求解时间
    #v2 求解超时时间
    #v3 大模型求解时间
    #v4 大模型求解是否成功
    time_out = 1200
    for k, v in info_dict.items():
        if v[1] > time_out:
            v[0] = 'unknown'
            v[1] = time_out
        if v[3] > time_out:
            v[4] = 'failed'
            v[3] = time_out
    del_list = []
    for k,v in info_dict.items():
        #筛选变量个数
        if v[1] > 300 and len(var_count[k]) > 5:
            #收集求解时间 z3solver
            if v[0] == 'sat':
                time_dict[k] = v[1]
                z3_sat_time += v[1]
            else:
                time_dict[k] = -v[1]
                z3_failed_time += v[1]
            #收集其他求解时间
            if v[4] == 'succeed':
                time_dict_2[k] = v[3]
            else:
                time_dict_2[k] = -v[3]
            if v[0] == 'sat' and v[1] <= time_out:
                sat_count += 1
                if v[4] == 'succeed':
                    result_dict['succeed'].append(k)
                    sat2succeed += 1
                    result_dict['sat to succeed'].append(k)
                else:
                    result_dict['failed'].append(k)
                    sat2failed += 1
                    result_dict['sat to failed'].append(k)
            # elif v[1]>900:
            elif v[0] == 'unknown':
                # print(k,v)
                unknown_count += 1
                if v[4] == 'succeed':
                    result_dict['succeed'].append(k)
                    unknown2succeed += 1
                    result_dict['unknown to succeed'].append(k)
                else:
                    result_dict['failed'].append(k)
                    unknown2failed += 1
                    result_dict['unknown to failed'].append(k)
            # print(k,v[0],v[1],v[2],v[3],v[4])
            count += 1
            all_time_sat += v[1]
            all_time_rl += v[3]
            if v[4] == 'succeed':
                succeed_count += 1
                all_time_newsolver_succeed += new_solver_dict[k][1]
                all_time_sat_succeed += v[1]
                all_time_rl_succeed += v[3]
            elif v[4] == 'failed':
                failed_count += 1
                all_time_newsolver_failed += new_solver_dict[k][1]
                all_time_sat_failed += v[1]
                all_time_rl_failed += v[3]
        else:
            del_list.append(k)                      #删除变量个数小于5的测试数据
    for k in del_list:
        del info_dict[k]
    # logger.info(result_dict)
    print(f"Total Count: {count}, Succeed Count: {succeed_count}, Failed Count: {failed_count}, "
          f"All Time SAT: {all_time_sat},All Time SAT avg: {all_time_sat/count}  All Time RL: {all_time_rl},All Time RL avg:{all_time_rl/count}, "
          f"All Time SAT Succeed: {all_time_sat_succeed}, All Time SAT Succeed avg: {all_time_sat_succeed/succeed_count}, All Time RL Succeed: {all_time_rl_succeed},All Time RL Succeed avg: {all_time_rl_succeed/succeed_count}, "
          f"All Time SAT Failed: {all_time_sat_failed}, All Time SAT Failed avg: {all_time_sat_failed/failed_count},All Time RL Failed: {all_time_rl_failed},All Time RL Failed avg: {all_time_rl_failed/failed_count}")
    print(f"All Time New Solver Succeed: {all_time_newsolver_succeed,all_time_newsolver_succeed/succeed_count}, All Time New Solver Failed: {all_time_newsolver_failed,all_time_newsolver_failed/failed_count}, ")
    print(f"Unknown Count: {unknown_count}, SAT Count: {sat_count}, "
          f"Unknown to Succeed: {unknown2succeed}, Unknown to Failed: {unknown2failed}, "
          f"SAT to Succeed: {sat2succeed}, SAT to Failed: {sat2failed}")
    print(result_dict)
    print(time_dict)
    return result_dict,time_dict,time_dict_2,info_dict
    #修改求解时间,求解时间大于1200s的测试数据记为unknown
def test_group_2_no_save_1207_only_llm(solve_name,info_name):
    time_dict = {}

    solve_dict = load_dictionary(solve_name)

    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1109_pre_SMTimer_llama3.1:70b_1200s.txt'
    var_count = load_dictionary('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/var_count.txt')
    info_dict = load_dictionary(info_name)
    for k,v in info_dict.items():
        if k in solve_dict.keys():
            info_dict[k][0] = solve_dict[k][0]
            info_dict[k][1] = solve_dict[k][1]
            info_dict[k][2] = solve_dict[k][2]
    # if not os.path.exists(info_name):
    #     # 文件不存在时，创建文件
    #     info_dict = {}
    #     with open(info_name, 'w') as file:
    #         json.dump(info_dict, file, indent=4)
    #     print(f'文件{info_name} 已创建。')
    # else:
    #     info_dict = load_dictionary(info_name)
    #     print(f'文件已存在。')
    result_dict = {}
    result_dict['succeed'] = []
    result_dict['failed'] = []
    result_dict['unknown to succeed'] = []
    result_dict['unknown to failed'] = []
    result_dict['sat to succeed'] = []
    result_dict['sat to failed'] = []
    count = 0
    succeed_count = 0
    failed_count = 0
    all_time_sat_succeed = 0
    all_time_rl_succeed = 0
    all_time_sat_failed = 0
    all_time_rl_failed = 0
    all_time_sat = 0
    all_time_rl = 0
    unknown_count = 0
    sat_count = 0
    unknown2succeed = 0
    unknown2failed = 0
    sat2succeed = 0
    sat2failed = 0


    #v0 原始求解是否成功
    #v1 实际求解时间
    #v2 求解超时时间
    #v3 大模型求解时间
    #v4 大模型求解是否成功
    time_out = 1200
    for k, v in info_dict.items():
        if v[1] > time_out:
            v[0] = 'unknown'
            v[1] = time_out
        if v[4] > time_out:
            v[3] = 'failed'
            v[4] = time_out
    for k,v in info_dict.items():
        #筛选变量个数
        if v[1] > 300 and len(var_count[k]) > 5:
            if v[3] == 'succeed':
                time_dict[k] = v[4]
            else:
                time_dict[k] = -v[4]
            if v[0] == 'sat' and v[1] <= time_out:
                sat_count += 1
                if v[3] == 'succeed':
                    result_dict['succeed'].append(k)
                    sat2succeed += 1
                    result_dict['sat to succeed'].append(k)
                else:
                    result_dict['failed'].append(k)
                    sat2failed += 1
                    result_dict['sat to failed'].append(k)
            # elif v[1]>900:
            elif v[0] == 'unknown':
                # print(k,v)
                unknown_count += 1
                if v[3] == 'succeed':
                    result_dict['succeed'].append(k)
                    unknown2succeed += 1
                    result_dict['unknown to succeed'].append(k)
                else:
                    result_dict['failed'].append(k)
                    unknown2failed += 1
                    result_dict['unknown to failed'].append(k)
            # print(k,v[0],v[1],v[2],v[3],v[4])
            count += 1
            all_time_sat += v[1]
            all_time_rl += v[4]
            if v[3] == 'succeed':
                succeed_count += 1
                all_time_sat_succeed += v[1]
                all_time_rl_succeed += v[4]
            elif v[3] == 'failed':
                failed_count += 1
                all_time_sat_failed += v[1]
                all_time_rl_failed += v[4]
    # logger.info(result_dict)
    print(f"Total Count: {count}, Succeed Count: {succeed_count}, Failed Count: {failed_count}, "
          f"All Time SAT: {all_time_sat},All Time SAT avg: {all_time_sat/count}  All Time RL: {all_time_rl},All Time RL avg:{all_time_rl/count}, "
          f"All Time SAT Succeed: {all_time_sat_succeed}, All Time SAT Succeed avg: {all_time_sat_succeed/succeed_count}, All Time RL Succeed: {all_time_rl_succeed},All Time RL Succeed avg: {all_time_rl_succeed/succeed_count}, "
          f"All Time SAT Failed: {all_time_sat_failed}, All Time SAT Failed avg: {all_time_sat_failed/failed_count},All Time RL Failed: {all_time_rl_failed},All Time RL Failed avg: {all_time_rl_failed/failed_count}")
    print(f"Unknown Count: {unknown_count}, SAT Count: {sat_count}, "
          f"Unknown to Succeed: {unknown2succeed}, Unknown to Failed: {unknown2failed}, "
          f"SAT to Succeed: {sat2succeed}, SAT to Failed: {sat2failed}")
    # print(f"z3_sat_time: {z3_sat_time}, z3_failed_time: {z3_failed_time}")
    # print(result_dict)
    return result_dict,time_dict
    #修改求解时间,求解时间大于1200s的测试数据记为unknown
def test_group_2_no_save_QF_IDL_0429(solve_name,info_name):
    time_dict = {}
    time_dict_2 = {}
    solve_dict = load_dictionary(solve_name)

    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1109_pre_SMTimer_llama3.1:70b_1200s.txt'

    info_dict = load_dictionary(info_name)
    for k,v in info_dict.items():
        if k in solve_dict.keys():
            info_dict[k][0] = solve_dict[k][0]
            info_dict[k][1] = solve_dict[k][1]
            info_dict[k][2] = solve_dict[k][2]
    # if not os.path.exists(info_name):
    #     # 文件不存在时，创建文件
    #     info_dict = {}
    #     with open(info_name, 'w') as file:
    #         json.dump(info_dict, file, indent=4)
    #     print(f'文件{info_name} 已创建。')
    # else:
    #     info_dict = load_dictionary(info_name)
    #     print(f'文件已存在。')
    result_dict = {}
    result_dict['succeed'] = []
    result_dict['failed'] = []
    result_dict['unknown to succeed'] = []
    result_dict['unknown to failed'] = []
    result_dict['sat to succeed'] = []
    result_dict['sat to failed'] = []
    count = 0
    succeed_count = 0
    failed_count = 0
    all_time_sat_succeed = 0
    all_time_rl_succeed = 0
    all_time_sat_failed = 0
    all_time_rl_failed = 0
    all_time_sat = 0
    all_time_rl = 0
    unknown_count = 0
    sat_count = 0
    unknown2succeed = 0
    unknown2failed = 0
    sat2succeed = 0
    sat2failed = 0
    z3_sat_time = 0
    z3_failed_time = 0
    #v0 原始求解是否成功
    #v1 实际求解时间
    #v2 求解超时时间
    #v3 大模型求解时间
    #v4 大模型求解是否成功
    time_out = 1200
    for k, v in info_dict.items():
        if v[1] > time_out:
            v[0] = 'unknown'
            v[1] = time_out
        if v[3] > time_out:
            v[4] = 'failed'
            v[3] = time_out
    del_list = []
    for k,v in info_dict.items():
        #筛选变量个数
        if v[1] > 300:
            #收集求解时间 z3solver
            if v[0] == 'sat':
                time_dict[k] = v[1]
                z3_sat_time += v[1]
            else:
                time_dict[k] = -v[1]
                z3_failed_time += v[1]
            #收集其他求解时间
            if v[4] == 'succeed':
                time_dict_2[k] = v[3]
            else:
                time_dict_2[k] = -v[3]
            if v[0] == 'sat' and v[1] <= time_out:
                sat_count += 1
                if v[4] == 'succeed':
                    result_dict['succeed'].append(k)
                    sat2succeed += 1
                    result_dict['sat to succeed'].append(k)
                else:
                    result_dict['failed'].append(k)
                    sat2failed += 1
                    result_dict['sat to failed'].append(k)
            # elif v[1]>900:
            elif v[0] == 'unknown':
                # print(k,v)
                unknown_count += 1
                if v[4] == 'succeed':
                    result_dict['succeed'].append(k)
                    unknown2succeed += 1
                    result_dict['unknown to succeed'].append(k)
                else:
                    result_dict['failed'].append(k)
                    unknown2failed += 1
                    result_dict['unknown to failed'].append(k)
            # print(k,v[0],v[1],v[2],v[3],v[4])
            count += 1
            all_time_sat += v[1]
            all_time_rl += v[3]
            if v[4] == 'succeed':
                succeed_count += 1
                all_time_sat_succeed += v[1]
                all_time_rl_succeed += v[3]
            elif v[4] == 'failed':
                failed_count += 1
                all_time_sat_failed += v[1]
                all_time_rl_failed += v[3]
        else:
            del_list.append(k)                      #删除变量个数小于5的测试数据
    for k in del_list:
        del info_dict[k]
    # logger.info(result_dict)
    print(f"Total Count: {count}, Succeed Count: {succeed_count}, Failed Count: {failed_count}, "
          f"All Time SAT: {all_time_sat},All Time SAT avg: {all_time_sat/count}  All Time RL: {all_time_rl},All Time RL avg:{all_time_rl/count}, "
          f"All Time SAT Succeed: {all_time_sat_succeed}, All Time SAT Succeed avg: {all_time_sat_succeed/succeed_count}, All Time RL Succeed: {all_time_rl_succeed},All Time RL Succeed avg: {all_time_rl_succeed/succeed_count}, "
          f"All Time SAT Failed: {all_time_sat_failed}, All Time SAT Failed avg: {all_time_sat_failed/failed_count},All Time RL Failed: {all_time_rl_failed},All Time RL Failed avg: {all_time_rl_failed/failed_count}")
    print(f"Unknown Count: {unknown_count}, SAT Count: {sat_count}, "
          f"Unknown to Succeed: {unknown2succeed}, Unknown to Failed: {unknown2failed}, "
          f"SAT to Succeed: {sat2succeed}, SAT to Failed: {sat2failed}")
    print(result_dict)
    print(time_dict)
    return result_dict,time_dict,time_dict_2,info_dict
    #修改求解时间,求解时间大于1200s的测试数据记为unknown
def spilt_class(data_dict):
    result_dict ={}
    new_dict = {}
    for k,v in data_dict.items():
        k_list = k.split('/')
        print(k_list)
        soft_class = k_list[-2]
        if soft_class not in new_dict.keys():
            new_dict[soft_class] = {}
            new_dict[soft_class][k] = v
        else:
            new_dict[soft_class][k] = v
    for k,v in new_dict.items():
        result_dict[k] = {}
        result_dict[k]['succeed'] = 0
        result_dict[k]['failed'] = 0
        result_dict[k]['succeed_time'] = 0
        result_dict[k]['failed_time'] = 0
        result_dict[k]['succeed_time_avg'] = 0
        result_dict[k]['all_time'] = 0
        result_dict[k]['avg'] = 0
        for k1,v1 in v.items():
            if v1>0:
                result_dict[k]['succeed'] += 1
                result_dict[k]['succeed_time'] += v1
                result_dict[k]['all_time'] += v1
            else:
                result_dict[k]['failed'] += 1
                result_dict[k]['failed_time'] += -v1
                result_dict[k]['all_time'] += -v1
        result_dict[k]['avg'] = result_dict[k]['all_time']/(result_dict[k]['succeed']+result_dict[k]['failed'])
        if result_dict[k]['succeed'] > 0:
            result_dict[k]['succeed_time_avg'] = result_dict[k]['succeed_time']/result_dict[k]['succeed']
    return new_dict,result_dict
def resolve_dataset():
    result_dict = {}
    result_dict_2 = {}
    solve_dict = load_dictionary('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_bingxing.txt')
    for k, v in solve_dict.items():
        if v[0] =='sat' and v[1] > 1200:
            solve_dict[k][0] = 'unknown'
            solve_dict[k][1] = 1200
        if v[0] == 'unknown' and v[1] > 1200:
            # solve_dict[k][0] = 'unknown'
            solve_dict[k][1] = 1200
    for k,v in solve_dict.items():
        k_list = k.split('/')
        if k_list[5] not in result_dict.keys():
            result_dict[k_list[5]]= {}
        if k_list[-2] not in result_dict[k_list[5]].keys():
            result_dict[k_list[5]][k_list[-2]] = {}
            result_dict[k_list[5]][k_list[-2]][k] = v
        else:
            result_dict[k_list[5]][k_list[-2]][k] = v
    # print(result_dict)
    for k,v in result_dict.items():
        result_dict_2[k] = {}
        for k1,v1 in v.items():
            result_dict_2[k][k1] = {}
            result_dict_2[k][k1]['sat'] = 0
            result_dict_2[k][k1]['unsat'] = 0
            result_dict_2[k][k1]['unknown'] = 0
            result_dict_2[k][k1]['sat_time'] = 0
            result_dict_2[k][k1]['unsat_time'] = 0
            result_dict_2[k][k1]['unknown_time'] = 0
            result_dict_2[k][k1]['sat+unknown_time'] = 0
            result_dict_2[k][k1]['sat_time_avg'] = 0
            result_dict_2[k][k1]['unsat_time_avg'] = 0
            result_dict_2[k][k1]['unknown_time_avg'] = 0
            result_dict_2[k][k1]['sat+unknown_time_avg'] = 0

            result_dict_2[k][k1]['all_time'] = 0
            result_dict_2[k][k1]['avg'] = 0
            #unsat数据



            for k2, v2 in v1.items():
                print(k2,v2)
                if v2[0] == 'sat':
                    result_dict_2[k][k1]['sat'] += 1
                    result_dict_2[k][k1]['sat_time'] += v2[1]
                    result_dict_2[k][k1]['all_time'] += v2[1]
                    result_dict_2[k][k1]['sat+unknown_time'] += v2[1]
                elif v2[0] == 'unknown':
                    result_dict_2[k][k1]['unknown'] += 1
                    result_dict_2[k][k1]['unknown_time'] += v2[1]
                    result_dict_2[k][k1]['all_time'] += v2[1]
                    result_dict_2[k][k1]['sat+unknown_time'] += v2[1]
                elif v2[0] == 'unsat':
                    result_dict_2[k][k1]['unsat'] += 1
                    result_dict_2[k][k1]['unsat_time'] += v2[1]
                    result_dict_2[k][k1]['all_time'] += v2[1]
            if result_dict_2[k][k1]['sat'] + result_dict_2[k][k1]['unknown'] + result_dict_2[k][k1]['unsat']> 0:
                result_dict_2[k][k1]['avg'] = result_dict_2[k][k1]['all_time'] / (result_dict_2[k][k1]['sat'] + result_dict_2[k][k1]['unknown'] + result_dict_2[k][k1]['unsat'])
            if result_dict_2[k][k1]['sat'] > 0:
                result_dict_2[k][k1]['sat_time_avg'] = result_dict_2[k][k1]['sat_time'] / result_dict_2[k][k1]['sat']
            if result_dict_2[k][k1]['sat'] + result_dict_2[k][k1]['unknown'] > 0:
                result_dict_2[k][k1]['sat+unknown_time_avg'] = result_dict_2[k][k1]['sat+unknown_time'] / (result_dict_2[k][k1]['sat'] + result_dict_2[k][k1]['unknown'])
            if result_dict_2[k][k1]['unsat'] > 0:
                result_dict_2[k][k1]['unsat_time_avg'] = result_dict_2[k][k1]['unsat_time'] / result_dict_2[k][k1]['unsat']
            if result_dict_2[k][k1]['unknown'] > 0:
                result_dict_2[k][k1]['unknown_time_avg'] = result_dict_2[k][k1]['unknown_time'] / result_dict_2[k][k1]['unknown']
    print(result_dict_2)
    for k,v in result_dict_2.items():
        for k1,v1 in v.items():
            print(k,k1,v1)
    with open('result_dict_z3solver.txt', 'w') as file:
        json.dump(result_dict_2, file, indent=4)
def resolve_dataset_300s():
    var_count = load_dictionary('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/var_count.txt')
    with open('/home/lz/sibyl_3/src/networks/info_dict_rl.txt', 'r') as file:
        rl_dict = json.load(file)
    result_dict = {}
    result_dict_2 = {}
    solve_dict = load_dictionary('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_bingxing.txt')
    for k, v in solve_dict.items():
        if v[0] =='sat' and v[1] > 1200:
            solve_dict[k][0] = 'unknown'
            solve_dict[k][1] = 1200
        if v[0] == 'unknown' and v[1] > 1200:
            # solve_dict[k][0] = 'unknown'
            solve_dict[k][1] = 1200
    for k,v in solve_dict.items():
        if (v[0] == "sat" or v[0] == "unknown") and v[1] > 300 and k in rl_dict.keys() and len(var_count[k]) > 5:
            k_list = k.split('/')
            if k_list[5] not in result_dict.keys():
                result_dict[k_list[5]]= {}
            if k_list[-2] not in result_dict[k_list[5]].keys():
                result_dict[k_list[5]][k_list[-2]] = {}
                result_dict[k_list[5]][k_list[-2]][k] = v
            else:
                result_dict[k_list[5]][k_list[-2]][k] = v
    # print(result_dict)
    for k,v in result_dict.items():
        result_dict_2[k] = {}
        for k1,v1 in v.items():
            result_dict_2[k][k1] = {}
            result_dict_2[k][k1]['sat'] = 0
            result_dict_2[k][k1]['unsat'] = 0
            result_dict_2[k][k1]['unknown'] = 0
            result_dict_2[k][k1]['sat_time'] = 0
            result_dict_2[k][k1]['unsat_time'] = 0
            result_dict_2[k][k1]['unknown_time'] = 0
            result_dict_2[k][k1]['sat+unknown_time'] = 0
            result_dict_2[k][k1]['sat_time_avg'] = 0
            result_dict_2[k][k1]['unsat_time_avg'] = 0
            result_dict_2[k][k1]['unknown_time_avg'] = 0
            result_dict_2[k][k1]['sat+unknown_time_avg'] = 0

            result_dict_2[k][k1]['all_time'] = 0
            result_dict_2[k][k1]['avg'] = 0
            #unsat数据
            result_dict_2[k][k1]['sat_list'] = []


            for k2, v2 in v1.items():
                print(k2,v2)
                if v2[0] == 'sat':
                    result_dict_2[k][k1]['sat'] += 1
                    result_dict_2[k][k1]['sat_time'] += v2[1]
                    result_dict_2[k][k1]['all_time'] += v2[1]
                    result_dict_2[k][k1]['sat+unknown_time'] += v2[1]

                    result_dict_2[k][k1]['sat_list'].append(v2[1])

                elif v2[0] == 'unknown':
                    result_dict_2[k][k1]['unknown'] += 1
                    result_dict_2[k][k1]['unknown_time'] += v2[1]
                    result_dict_2[k][k1]['all_time'] += v2[1]
                    result_dict_2[k][k1]['sat+unknown_time'] += v2[1]
                elif v2[0] == 'unsat':
                    result_dict_2[k][k1]['unsat'] += 1
                    result_dict_2[k][k1]['unsat_time'] += v2[1]
                    result_dict_2[k][k1]['all_time'] += v2[1]
            if result_dict_2[k][k1]['sat'] + result_dict_2[k][k1]['unknown'] + result_dict_2[k][k1]['unsat']> 0:
                result_dict_2[k][k1]['avg'] = result_dict_2[k][k1]['all_time'] / (result_dict_2[k][k1]['sat'] + result_dict_2[k][k1]['unknown'] + result_dict_2[k][k1]['unsat'])
            if result_dict_2[k][k1]['sat'] > 0:
                result_dict_2[k][k1]['sat_time_avg'] = result_dict_2[k][k1]['sat_time'] / result_dict_2[k][k1]['sat']
            if result_dict_2[k][k1]['sat'] + result_dict_2[k][k1]['unknown'] > 0:
                result_dict_2[k][k1]['sat+unknown_time_avg'] = result_dict_2[k][k1]['sat+unknown_time'] / (result_dict_2[k][k1]['sat'] + result_dict_2[k][k1]['unknown'])
            if result_dict_2[k][k1]['unsat'] > 0:
                result_dict_2[k][k1]['unsat_time_avg'] = result_dict_2[k][k1]['unsat_time'] / result_dict_2[k][k1]['unsat']
            if result_dict_2[k][k1]['unknown'] > 0:
                result_dict_2[k][k1]['unknown_time_avg'] = result_dict_2[k][k1]['unknown_time'] / result_dict_2[k][k1]['unknown']
    print(result_dict_2)
    for k,v in result_dict_2.items():
        for k1,v1 in v.items():
            print(k,k1,v1)
    with open('result_dict_z3solver_300s.txt', 'w') as file:
        json.dump(result_dict_2, file, indent=4)
def resolve_dataset_cvc5_smtimer(file_path='/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/cvc5_smtimer_results.json'):
    result_dict = {}
    result_dict_2 = {}
    solve_dict = load_dictionary(file_path)
    for k, v in solve_dict.items():
        if v[0] =='sat' and v[1] > 1200:
            solve_dict[k][0] = 'unknown'
            solve_dict[k][1] = 1200
        if (v[0] == 'unknown' or v[0] == 'timeout') and v[1] >= 1200:
            solve_dict[k][0] = 'unknown'
            solve_dict[k][1] = 1200
    for k,v in solve_dict.items():
        k_list = k.split('/')
        if k_list[5] not in result_dict.keys():
            result_dict[k_list[5]]= {}
        if k_list[-2] not in result_dict[k_list[5]].keys():
            result_dict[k_list[5]][k_list[-2]] = {}
            result_dict[k_list[5]][k_list[-2]][k] = v
        else:
            result_dict[k_list[5]][k_list[-2]][k] = v
    # print(result_dict)
    for k,v in result_dict.items():
        result_dict_2[k] = {}
        for k1,v1 in v.items():
            result_dict_2[k][k1] = {}
            result_dict_2[k][k1]['sat'] = 0
            result_dict_2[k][k1]['unsat'] = 0
            result_dict_2[k][k1]['unknown'] = 0
            result_dict_2[k][k1]['sat_time'] = 0
            result_dict_2[k][k1]['unsat_time'] = 0
            result_dict_2[k][k1]['unknown_time'] = 0
            result_dict_2[k][k1]['sat+unknown_time'] = 0
            result_dict_2[k][k1]['sat_time_avg'] = 0
            result_dict_2[k][k1]['unsat_time_avg'] = 0
            result_dict_2[k][k1]['unknown_time_avg'] = 0
            result_dict_2[k][k1]['sat+unknown_time_avg'] = 0

            result_dict_2[k][k1]['all_time'] = 0
            result_dict_2[k][k1]['avg'] = 0
            #unsat数据



            for k2, v2 in v1.items():
                print(k2,v2)
                if v2[0] == 'sat':
                    result_dict_2[k][k1]['sat'] += 1
                    result_dict_2[k][k1]['sat_time'] += v2[1]
                    result_dict_2[k][k1]['all_time'] += v2[1]
                    result_dict_2[k][k1]['sat+unknown_time'] += v2[1]
                elif v2[0] == 'unknown':
                    result_dict_2[k][k1]['unknown'] += 1
                    result_dict_2[k][k1]['unknown_time'] += v2[1]
                    result_dict_2[k][k1]['all_time'] += v2[1]
                    result_dict_2[k][k1]['sat+unknown_time'] += v2[1]
                elif v2[0] == 'unsat':
                    result_dict_2[k][k1]['unsat'] += 1
                    result_dict_2[k][k1]['unsat_time'] += v2[1]
                    result_dict_2[k][k1]['all_time'] += v2[1]
            if result_dict_2[k][k1]['sat'] + result_dict_2[k][k1]['unknown'] + result_dict_2[k][k1]['unsat']> 0:
                result_dict_2[k][k1]['avg'] = result_dict_2[k][k1]['all_time'] / (result_dict_2[k][k1]['sat'] + result_dict_2[k][k1]['unknown'] + result_dict_2[k][k1]['unsat'])
            if result_dict_2[k][k1]['sat'] > 0:
                result_dict_2[k][k1]['sat_time_avg'] = result_dict_2[k][k1]['sat_time'] / result_dict_2[k][k1]['sat']
            if result_dict_2[k][k1]['sat'] + result_dict_2[k][k1]['unknown'] > 0:
                result_dict_2[k][k1]['sat+unknown_time_avg'] = result_dict_2[k][k1]['sat+unknown_time'] / (result_dict_2[k][k1]['sat'] + result_dict_2[k][k1]['unknown'])
            if result_dict_2[k][k1]['unsat'] > 0:
                result_dict_2[k][k1]['unsat_time_avg'] = result_dict_2[k][k1]['unsat_time'] / result_dict_2[k][k1]['unsat']
            if result_dict_2[k][k1]['unknown'] > 0:
                result_dict_2[k][k1]['unknown_time_avg'] = result_dict_2[k][k1]['unknown_time'] / result_dict_2[k][k1]['unknown']
    print(result_dict_2)
    for k,v in result_dict_2.items():
        for k1,v1 in v.items():
            print(k,k1,v1)
    sat_count = 0
    sat_time = 0
    unsat_count = 0
    unsat_time = 0
    unknown_count = 0
    unknown_time = 0
    var_count = load_dictionary('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/var_count.txt')
    for k,v in solve_dict.items():
        if len(var_count[k]) > 5:
            if v[0] == 'sat':
                sat_count += 1
                sat_time += v[1]
            elif v[0] == 'unsat':
                unsat_count += 1
                unsat_time += v[1]
            elif v[0] == 'unknown':
                unknown_count += 1
                unknown_time += v[1]
    print(f"Total Count: {len(solve_dict)}, SAT Count: {sat_count}, UNSAT Count: {unsat_count}, UNKNOWN Count: {unknown_count}")
    print(f"SAT Time: {sat_time}, UNSAT Time: {unsat_time}, UNKNOWN Time: {unknown_time}")
    print(f"SAT Time Avg: {sat_time/sat_count if sat_count > 0 else 0}, UNSAT Time Avg: {unsat_time/unsat_count if unsat_count > 0 else 0}, UNKNOWN Time Avg: {unknown_time/unknown_count if unknown_count > 0 else 0}")
    print(
        f"failed Time Avg: {(unsat_time+unknown_time) / (unsat_count + unknown_count) if (unsat_count + unknown_count) > 0 else 0}")
    print(f"sat+unknown Time Avg: {(sat_time + unknown_time) / (sat_count + unknown_count) if (sat_count + unknown_count) > 0 else 0}")
    print(f"all time {sat_time + unsat_time + unknown_time} all time avg {(sat_time + unsat_time + unknown_time) / (sat_count + unsat_count + unknown_count) if (sat_count + unsat_count + unknown_count) > 0 else 0}")
    # 修改求解时间,求解时间大于1200s的测试数据记为unknown
    # with open('result_dict_z3solver.txt', 'w') as file:
    #     json.dump(result_dict_2, file, indent=4)
def resolve_time_dict_300s(data_dict):

    result_dict = {}
    result_dict_2 = {}
    #收集每个类别下的求解时间

    for k,v in data_dict.items():
        k_list = k.split('/')
        if k_list[5] not in result_dict.keys():
            result_dict[k_list[5]]= {}
        if k_list[-2] not in result_dict[k_list[5]].keys():
            result_dict[k_list[5]][k_list[-2]] = {}
            result_dict[k_list[5]][k_list[-2]][k] = v
        else:
            result_dict[k_list[5]][k_list[-2]][k] = v
    # print(result_dict)
    for k,v in result_dict.items():
        result_dict_2[k] = {}
        for k1,v1 in v.items():
            result_dict_2[k][k1] = {}

            result_dict_2[k][k1]['sat'] = 0
            result_dict_2[k][k1]['unsat'] = 0
            result_dict_2[k][k1]['unknown'] = 0
            result_dict_2[k][k1]['sat_time'] = 0
            result_dict_2[k][k1]['unsat_time'] = 0
            result_dict_2[k][k1]['unknown_time'] = 0
            result_dict_2[k][k1]['sat+unknown_time'] = 0
            result_dict_2[k][k1]['sat_time_avg'] = 0
            result_dict_2[k][k1]['unsat_time_avg'] = 0
            result_dict_2[k][k1]['unknown_time_avg'] = 0
            result_dict_2[k][k1]['sat+unknown_time_avg'] = 0

            result_dict_2[k][k1]['all_time'] = 0
            result_dict_2[k][k1]['avg'] = 0

            result_dict_2[k][k1]['sat_list'] = []
            #unsat数据



            for k2, v2 in v1.items():
                print(k2,v2)
                if v2[0] == 'sat':
                    result_dict_2[k][k1]['sat'] += 1
                    result_dict_2[k][k1]['sat_time'] += v2[1]
                    result_dict_2[k][k1]['all_time'] += v2[1]
                    result_dict_2[k][k1]['sat+unknown_time'] += v2[1]
                    #统计时间
                    result_dict_2[k][k1]['sat_list'].append(v2[1])
                elif v2[0] == 'unknown':
                    result_dict_2[k][k1]['unknown'] += 1
                    result_dict_2[k][k1]['unknown_time'] += v2[1]
                    result_dict_2[k][k1]['all_time'] += v2[1]
                    result_dict_2[k][k1]['sat+unknown_time'] += v2[1]
                elif v2[0] == 'unsat':
                    result_dict_2[k][k1]['unsat'] += 1
                    result_dict_2[k][k1]['unsat_time'] += v2[1]
                    result_dict_2[k][k1]['all_time'] += v2[1]
            if result_dict_2[k][k1]['sat'] + result_dict_2[k][k1]['unknown'] + result_dict_2[k][k1]['unsat']> 0:
                result_dict_2[k][k1]['avg'] = result_dict_2[k][k1]['all_time'] / (result_dict_2[k][k1]['sat'] + result_dict_2[k][k1]['unknown'] + result_dict_2[k][k1]['unsat'])
            if result_dict_2[k][k1]['sat'] > 0:
                result_dict_2[k][k1]['sat_time_avg'] = result_dict_2[k][k1]['sat_time'] / result_dict_2[k][k1]['sat']
            if result_dict_2[k][k1]['sat'] + result_dict_2[k][k1]['unknown'] > 0:
                result_dict_2[k][k1]['sat+unknown_time_avg'] = result_dict_2[k][k1]['sat+unknown_time'] / (result_dict_2[k][k1]['sat'] + result_dict_2[k][k1]['unknown'])
            if result_dict_2[k][k1]['unsat'] > 0:
                result_dict_2[k][k1]['unsat_time_avg'] = result_dict_2[k][k1]['unsat_time'] / result_dict_2[k][k1]['unsat']
            if result_dict_2[k][k1]['unknown'] > 0:
                result_dict_2[k][k1]['unknown_time_avg'] = result_dict_2[k][k1]['unknown_time'] / result_dict_2[k][k1]['unknown']
    print(result_dict_2)
    for k,v in result_dict_2.items():
        for k1,v1 in v.items():
            print(k,k1,v1)
    return result_dict_2
    # with open('result_dict_z3solver_300s.txt', 'w') as file:
    #     json.dump(result_dict_2, file, indent=4)
def resolve_time_dict_300s_RL_LLM(data_dict):
    var_count = load_dictionary('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/var_count.txt')
    with open('/home/lz/sibyl_3/src/networks/info_dict_rl.txt', 'r') as file:
        rl_dict = json.load(file)
    result_dict = {}
    result_dict_2 = {}
    #收集每个类别下的求解时间

    for k,v in data_dict.items():
        if (v[0] == "sat" or v[0] == "unknown") and v[1] > 300 and k in rl_dict.keys() and len(var_count[k]) > 5:
            k_list = k.split('/')
            if k_list[5] not in result_dict.keys():
                result_dict[k_list[5]]= {}
            if k_list[-2] not in result_dict[k_list[5]].keys():
                result_dict[k_list[5]][k_list[-2]] = {}
                result_dict[k_list[5]][k_list[-2]][k] = v
            else:
                result_dict[k_list[5]][k_list[-2]][k] = v
    # print(result_dict)
    for k,v in result_dict.items():
        result_dict_2[k] = {}
        for k1,v1 in v.items():
            result_dict_2[k][k1] = {}

            result_dict_2[k][k1]['sat'] = 0
            result_dict_2[k][k1]['unsat'] = 0
            result_dict_2[k][k1]['unknown'] = 0
            result_dict_2[k][k1]['sat_time'] = 0
            result_dict_2[k][k1]['unsat_time'] = 0
            result_dict_2[k][k1]['unknown_time'] = 0
            result_dict_2[k][k1]['sat+unknown_time'] = 0
            result_dict_2[k][k1]['sat_time_avg'] = 0
            result_dict_2[k][k1]['unsat_time_avg'] = 0
            result_dict_2[k][k1]['unknown_time_avg'] = 0
            result_dict_2[k][k1]['sat+unknown_time_avg'] = 0

            result_dict_2[k][k1]['all_time'] = 0
            result_dict_2[k][k1]['avg'] = 0

            result_dict_2[k][k1]['sat_list'] = []
            #unsat数据



            for k2, v2 in v1.items():
                print(k2,v2)
                if v2[4] == 'succeed':
                    result_dict_2[k][k1]['sat'] += 1
                    result_dict_2[k][k1]['sat_time'] += v2[3]
                    result_dict_2[k][k1]['all_time'] += v2[3]
                    result_dict_2[k][k1]['sat+unknown_time'] += v2[3]
                    #统计时间
                    result_dict_2[k][k1]['sat_list'].append(v2[3])
                elif v2[4] == 'failed':
                    result_dict_2[k][k1]['unknown'] += 1
                    result_dict_2[k][k1]['unknown_time'] += v2[3]
                    result_dict_2[k][k1]['all_time'] += v2[3]
                    result_dict_2[k][k1]['sat+unknown_time'] += v2[3]
                # elif v2[0] == 'unsat':
                #     result_dict_2[k][k1]['unsat'] += 1
                #     result_dict_2[k][k1]['unsat_time'] += v2[1]
                #     result_dict_2[k][k1]['all_time'] += v2[1]
            if result_dict_2[k][k1]['sat'] + result_dict_2[k][k1]['unknown'] + result_dict_2[k][k1]['unsat']> 0:
                result_dict_2[k][k1]['avg'] = result_dict_2[k][k1]['all_time'] / (result_dict_2[k][k1]['sat'] + result_dict_2[k][k1]['unknown'] + result_dict_2[k][k1]['unsat'])
            if result_dict_2[k][k1]['sat'] > 0:
                result_dict_2[k][k1]['sat_time_avg'] = result_dict_2[k][k1]['sat_time'] / result_dict_2[k][k1]['sat']
            if result_dict_2[k][k1]['sat'] + result_dict_2[k][k1]['unknown'] > 0:
                result_dict_2[k][k1]['sat+unknown_time_avg'] = result_dict_2[k][k1]['sat+unknown_time'] / (result_dict_2[k][k1]['sat'] + result_dict_2[k][k1]['unknown'])
            if result_dict_2[k][k1]['unsat'] > 0:
                result_dict_2[k][k1]['unsat_time_avg'] = result_dict_2[k][k1]['unsat_time'] / result_dict_2[k][k1]['unsat']
            if result_dict_2[k][k1]['unknown'] > 0:
                result_dict_2[k][k1]['unknown_time_avg'] = result_dict_2[k][k1]['unknown_time'] / result_dict_2[k][k1]['unknown']
    print(result_dict_2)
    for k,v in result_dict_2.items():
        for k1,v1 in v.items():
            print(k,k1,v1)
    return result_dict_2
    # with open('result_dict_z3solver_300s.txt', 'w') as file:
    #     json.dump(result_dict_2, file, indent=4)
#smt-timer
def resolve_dataset_by_var_count():
    result_dict = {}
    result_dict_2 = {}
    with open('/home/lz/sibyl_3/src/networks/info_dict_rl.txt', 'r') as file:
        rl_dict = json.load(file)
    solve_dict = load_dictionary('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_bingxing.txt')
    for k, v in solve_dict.items():
        if v[0] =='sat' and v[1] > 1200:
            solve_dict[k][0] = 'unknown'
            solve_dict[k][1] = 1200
        if v[0] == 'unknown' and v[1] > 1200:
            # solve_dict[k][0] = 'unknown'
            solve_dict[k][1] = 1200
    for k,v in solve_dict.items():
        list1 = v
        if list1[0] == "sat" or list1[0] == "unknown":
            if list1[1] > 300 and k in rl_dict.keys():
                # if '/who/who86404' in key:
                print(k, v)
                file_path = k
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
                if file_path not in result_dict.keys():
                    print(type(smtlib_str))
                    smtlib_str, var_dict, constant_list = normalize_smt_str(smtlib_str)
                    print(var_dict)
                    result_dict[file_path] = var_dict
                    with open('var_count.txt', 'w') as file:
                        json.dump(result_dict, file, indent=4)


def get_var_count_from_dataset(input_path, output_path):
    """
    从数据集中统计变量个数
    :param input_path: 输入文件路径，包含待处理文件路径的字典
    :param output_path: 输出文件路径，保存变量统计结果
    """
    input_dict = load_dictionary(input_path)
    result_dict = {}

    for file_path in input_dict.keys():
        smtlib_str = ''
        try:
            if not os.path.exists(file_path):
                print(f"文件未找到: {file_path}, 跳过。")
                continue

            if file_path.endswith('.smt2'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    smtlib_str = f.read()
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    dict_obj = json.loads(content)
                    if 'smt-comp' in file_path:
                        smtlib_str = dict_obj['smt_script']
                    else:
                        smtlib_str = dict_obj['script']
                except json.JSONDecodeError:
                    print(f"非 .smt2 文件的 JSON 解码错误: {file_path}, 跳过。")
                    continue

        except FileNotFoundError:
            print(f"文件未找到: {file_path}, 跳过。")
            continue
        except Exception as e:
            print(f"处理 {file_path} 时发生错误: {e}")
            continue

        if smtlib_str:
            if file_path not in result_dict:
                _, var_dict, _ = normalize_smt_str(smtlib_str)
                result_dict[file_path] = var_dict
        else:
            print(f"无法从 {file_path} 中提取 SMT 脚本, 跳过。")

    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=4)
    
    print(f"变量统计完成。结果已保存到 {output_path}")

def test_group_cvc5_process_analysis(info_name, var_count_path='/home/lz/PycharmProjects/Pearl/test_rl/test_solve/var_count.txt',
                                   output_dir=None, solver_name="CVC5"):
    """
    分析CVC5处理结果的方法，类似于test_group_2_no_save_1207
    专门处理run_predictor.py生成的info_dict文件

    Args:
        info_name: CVC5处理结果文件路径
        var_count_path: 变量统计文件路径
        output_dir: SuperVenn图输出目录，如果为None则不保存图片
        solver_name: 求解器名称，用于图表标签

    Returns:
        result_dict: 分类结果字典
        time_dict: 原始求解器时间字典
        time_dict_2: RL+LLM时间字典
        info_dict: 处理后的完整信息字典
        supervenn_stats: SuperVenn统计信息字典
    """
    time_dict = {}
    time_dict_2 = {}

    # 加载变量统计文件
    var_count = load_dictionary(var_count_path)

    # 加载CVC5处理结果文件
    info_dict = load_dictionary(info_name)

    # 初始化结果分类字典
    result_dict = {}
    result_dict['succeed'] = []
    result_dict['failed'] = []
    result_dict['unknown to succeed'] = []
    result_dict['unknown to failed'] = []
    result_dict['sat to succeed'] = []
    result_dict['sat to failed'] = []

    # 初始化统计变量
    count = 0
    succeed_count = 0
    failed_count = 0
    all_time_sat_succeed = 0
    all_time_rl_succeed = 0
    all_time_sat_failed = 0
    all_time_rl_failed = 0
    all_time_sat = 0
    all_time_rl = 0
    unknown_count = 0
    sat_count = 0
    unknown2succeed = 0
    unknown2failed = 0
    sat2succeed = 0
    sat2failed = 0
    z3_sat_time = 0
    z3_failed_time = 0

    # CVC5处理结果数据结构说明:
    # v[0] - 原始求解结果 ("sat"/"unsat"/"unknown")
    # v[1] - 原始求解时间 (秒)
    # v[2] - 超时限制 (秒)
    # v[3] - 总执行时间 (秒)
    # v[4] - 总求解时间 (秒)
    # v[5] - 最终求解时间 (秒)
    # v[6] - LLM时间 (秒)
    # v[7] - RL+LLM结果状态 ("succeed"/"failed")
    # v[8] - 最终赋值序列 (列表)
    # v[9] - 所有反例集合 (列表的列表)

    time_out = 1200

    # 超时处理
    for k, v in info_dict.items():
        if len(v) >= 2 and v[1] > time_out:
            v[0] = 'unknown'
            v[1] = time_out
        if len(v) >= 4 and v[3] > time_out:
            v[7] = 'failed'  # 更新状态为失败
            v[3] = time_out

    # 筛选和统计
    del_list = []
    for k, v in info_dict.items():
        # 检查数据完整性
        if len(v) < 8:
            print(f"警告: 文件 {k} 的数据不完整，跳过")
            del_list.append(k)
            continue

        # 检查是否在变量统计中
        if k not in var_count:
            del_list.append(k)
            continue

        # 筛选条件：原始求解时间>300s 且 变量数量>5
        if v[1] > 300 and len(var_count[k]) > 5:
            # 收集原始求解器时间
            if v[0] == 'sat':
                time_dict[k] = v[1]
                z3_sat_time += v[1]
            else:
                time_dict[k] = -v[1]
                z3_failed_time += v[1]

            # 收集RL+LLM求解时间 (使用总执行时间)
            if v[7] == 'succeed':
                time_dict_2[k] = v[3]  # 总执行时间
            else:
                time_dict_2[k] = -v[3]  # 负值表示失败

            # 分类统计
            if v[0] == 'sat' and v[1] <= time_out:
                sat_count += 1
                if v[7] == 'succeed':
                    result_dict['succeed'].append(k)
                    sat2succeed += 1
                    result_dict['sat to succeed'].append(k)
                else:
                    result_dict['failed'].append(k)
                    sat2failed += 1
                    result_dict['sat to failed'].append(k)
            elif v[0] == 'unknown':
                unknown_count += 1
                if v[7] == 'succeed':
                    result_dict['succeed'].append(k)
                    unknown2succeed += 1
                    result_dict['unknown to succeed'].append(k)
                else:
                    result_dict['failed'].append(k)
                    unknown2failed += 1
                    result_dict['unknown to failed'].append(k)

            # 累计统计
            count += 1
            all_time_sat += v[1]
            all_time_rl += v[3]  # 使用总执行时间

            if v[7] == 'succeed':
                succeed_count += 1
                all_time_sat_succeed += v[1]
                all_time_rl_succeed += v[3]
            elif v[7] == 'failed':
                failed_count += 1
                all_time_sat_failed += v[1]
                all_time_rl_failed += v[3]
        else:
            del_list.append(k)  # 删除不符合条件的测试数据

    # 删除不符合条件的数据
    for k in del_list:
        if k in info_dict:
            del info_dict[k]

    # 输出统计结果
    print(f"Total Count: {count}, Succeed Count: {succeed_count}, Failed Count: {failed_count}, "
          f"All Time SAT: {all_time_sat}, All Time SAT avg: {all_time_sat/count if count > 0 else 0:.2f}, "
          f"All Time RL: {all_time_rl}, All Time RL avg: {all_time_rl/count if count > 0 else 0:.2f}, "
          f"All Time SAT Succeed: {all_time_sat_succeed}, "
          f"All Time SAT Succeed avg: {all_time_sat_succeed/succeed_count if succeed_count > 0 else 0:.2f}, "
          f"All Time RL Succeed: {all_time_rl_succeed}, "
          f"All Time RL Succeed avg: {all_time_rl_succeed/succeed_count if succeed_count > 0 else 0:.2f}, "
          f"All Time SAT Failed: {all_time_sat_failed}, "
          f"All Time SAT Failed avg: {all_time_sat_failed/failed_count if failed_count > 0 else 0:.2f}, "
          f"All Time RL Failed: {all_time_rl_failed}, "
          f"All Time RL Failed avg: {all_time_rl_failed/failed_count if failed_count > 0 else 0:.2f}")

    print(f"Unknown Count: {unknown_count}, SAT Count: {sat_count}, "
          f"Unknown to Succeed: {unknown2succeed}, Unknown to Failed: {unknown2failed}, "
          f"SAT to Succeed: {sat2succeed}, SAT to Failed: {sat2failed}")

    print("Result Dictionary:")
    for key, value in result_dict.items():
        print(f"  {key}: {len(value)} files")

    print("Time Dictionary (first 5 entries):")
    for i, (key, value) in enumerate(time_dict.items()):
        if i >= 5:
            break
        print(f"  {key}: {value}")

    # 生成SuperVenn图
    supervenn_stats = None
    if output_dir is not None:
        print(f"\nGenerating SuperVenn diagrams...")

        # 构建求解集合
        baseline_solved = set()
        rl_llm_solved = set()

        for k, v in info_dict.items():
            # 基线求解器解决的约束（sat或unsat）
            if v[0] in ['sat', 'unsat']:
                baseline_solved.add(k)

            # RL+LLM解决的约束
            if v[7] == 'succeed':
                rl_llm_solved.add(k)

        # 绘制SuperVenn图
        supervenn_stats = plot_supervenn_diagrams_cvc5(
            baseline_solved,
            rl_llm_solved,
            solver_name=solver_name,
            output_dir=output_dir
        )
    else:
        print("\nSkipping SuperVenn diagram generation (no output directory specified)")

    return result_dict, time_dict, time_dict_2, info_dict, supervenn_stats
if __name__ == '__main__':
    #获取变量个数 QF_NIA
    # get_var_count_from_dataset('/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_0503_pre_llm_llama3.1:70b_1200s_QF_NIA.txt', 'info_QF_NIA_var_count.txt')
    #获取变量个数 QF_LIA
    # get_var_count_from_dataset(
    #     '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1118_pre_llm_llama3.1:70b_1200s_QF_LIA.txt',
    #     'info_QF_LIA_var_count.txt')
    #获取变量个数 cvc5 smtimer
    # get_var_count_from_dataset('/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/cvc5_process/info_dict_SMTimer_llama3.1:70b_1200s_info_dict_rl_cvc5_0628.txt', 'cvc5_smtimer_var_count.txt')
    # test_group()
    # test_group_1()
    # test_group_2()
    # test_group_3()
    # test_group_1_save()
    # test_group_1_no_save()
    # test_group_1_llm()
    # something()
    #z3结果smtimer
    # resolve_dataset_cvc5_smtimer('/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/z3_smtimer_results.json')
    # #cvc5结果smtimer
    # resolve_dataset_cvc5_smtimer()
    #cvc5 llama3.1
    solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/cvc5_smtimer_results_rl.json'
    info_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/cvc5_process/info_dict_SMTimer_llama3.1:70b_1200s_info_dict_rl_cvc5_0628.txt'
    var_count_path = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/cvc5_smtimer_var_count.txt'
    # result_dict_1 = test_group_2_no_save(solve_name,info_name)
    # 示例调用，包含SuperVenn图生成
    output_dir = '/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/cvc5_process/supervenn_output'
    result_dict_2, time_dict, time_dict_2, info_dict, supervenn_stats = test_group_cvc5_process_analysis(
        info_name, var_count_path, output_dir=output_dir, solver_name="CVC5"
    )

    # 测试新的CVC5处理结果分析方法 (需要时取消注释)
    # print("\n" + "="*80)
    # print("CVC5 Process Analysis Results:")
    # print("="*80)
    # cvc5_result_dict, cvc5_time_dict, cvc5_time_dict_2, cvc5_info_dict, cvc5_supervenn_stats = test_group_cvc5_process_analysis(info_name, var_count_path, output_dir='/path/to/output', solver_name="CVC5")

    #mathsat结果smtimer
    # resolve_dataset_cvc5_smtimer('/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/mathsat5_smtimer_results.json')

    # QF_NIA llama3.1
    # solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/NIA/NIA.json'
    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_0503_pre_llm_llama3.1:70b_1200s_QF_NIA.txt'
    # # result_dict_1 = test_group_2_no_save(solve_name,info_name)
    # result_dict_2, time_dict, time_dict_2, info_dict = test_group_2_no_save_QF_IDL_0429(solve_name, info_name)

    #QF_LIA llama3.1
    # solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_smt_comp.txt'
    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1118_pre_llm_llama3.1:70b_1200s_QF_LIA.txt'
    # # result_dict_1 = test_group_2_no_save(solve_name,info_name)
    # result_dict_2,time_dict,time_dict_2,info_dict =test_group_2_no_save_QF_IDL_0429(solve_name, info_name)

    # #deepseekr1:70b 70b
    # solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_bingxing.txt'
    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_0107_pre_SMTimer_deepseek-r1:70b_1200s_info_dict_rl.txt'
    # # result_dict_1 = test_group_2_no_save(solve_name,info_name)
    # result_dict_2,time_dict,time_dict_2,info_dict =test_group_2_no_save_1207(solve_name, info_name)

    # #llama3.3 70b
    # solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_bingxing.txt'
    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_0107_pre_SMTimer_llama3.3:70b_1200s_info_dict_rl.txt'
    # # result_dict_1 = test_group_2_no_save(solve_name,info_name)
    # result_dict_2,time_dict,time_dict_2,info_dict =test_group_2_no_save_1207(solve_name, info_name)

    # RL+LLM CVC5
    # solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_bingxing.txt'
    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1110_pre_SMTimer_llama3.1:70b_1200s_info_dict_rl.txt'
    # new_solver = '/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/cvc5_smtimer_results.json'
    # result_dict_2, time_dict, time_dict_2, info_dict = test_group_2_no_save_0607_cvc5(solve_name, info_name, new_solver)
    #RL+LLM
    # solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_bingxing.txt'
    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1110_pre_SMTimer_llama3.1:70b_1200s_info_dict_rl.txt'
    # # result_dict_1 = test_group_2_no_save(solve_name,info_name)
    # result_dict_2,time_dict,time_dict_2,info_dict =test_group_2_no_save_1207(solve_name, info_name)
    # # with open('time_dict_z3solver_106.txt', 'w') as file:
    # #     json.dump(time_dict, file, indent=4)
    # # with open('time_dict_RL+LLM_106.txt', 'w') as file:
    # #     json.dump(time_dict_2, file, indent=4)
    # result_dict = resolve_time_dict_300s_RL_LLM(info_dict)
    # with open('result_dict_RL+LLM_108.txt', 'w') as file:
    #     json.dump(result_dict, file, indent=4)
    # new_dict, result_dict = spilt_class(time_dict_2)

    # #统计z3求解个数
    # sat_count = 0
    # unknown_count = 0
    # sat_time = 0
    # unknown_time = 0
    # for k,v in time_dict.items():
    #     if v>0：
    #        sat_count += 1
    #        sat_time += v
    #     else:
    #         unknown_count += 1
    # print(sat_count,unknown_count)
    # print(sat_time,sat_time/sat_count)

    #计算z3与RL+LLM的求解区别
    # list1 = []
    # for k,v in time_dict.items():
    #     if v>0:
    #         list1.append(k)
    # list2 = []
    # for k,v in time_dict_2.items():
    #     if v>0:
    #         list2.append(k)
    #
    # set1 = set(list1)
    # set2 = set(list2)
    # unique_to_set1 = set1 - set2
    # print(len(unique_to_set1))
    # print(unique_to_set1)
    # unique_to_set2 = set2 - set1
    # print(len(unique_to_set2))
    # print(unique_to_set2)
    # set1_dict={}
    # set2_dict={}
    # for i in unique_to_set1:
    #     i_list = i.split('/')
    #     print(i_list)
    #     if i_list[-2] not in set1_dict.keys():
    #         set1_dict[i_list[-2]] =[]
    #     set1_dict[i_list[-2]].append(i)
    #
    # for i in unique_to_set2:
    #     i_list = i.split('/')
    #     print(i_list)
    #     if i_list[-2] not in set2_dict.keys():
    #         set2_dict[i_list[-2]] =[]
    #     set2_dict[i_list[-2]].append(i)
    # # print(set1_dict)
    # # print(set2_dict)
    # for k,v in set1_dict.items():
    #     print(k,v)
    # print('-------------------')
    # for k,v in set2_dict.items():
    #     print(k,v)
    # Random+LLM
    # solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_bingxing.txt'
    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1217_pre_SMTimer_llama3.1:70b_1200s_info_dict_rl_random.txt'
    # # result_dict_1 = test_group_2_no_save(solve_name,info_name)
    # result_dict_2,time_dict,time_dict_2,info_dict =test_group_2_no_save_1207(solve_name, info_name)
    # print(result_dict_2)
    # result_dict = resolve_time_dict_300s(info_dict)
    # with open('result_dict_RL+LLM_108.txt', 'w') as file:
    #     json.dump(result_dict, file, indent=4)
    # # print(time_dict)
    # # new_dict,result_dict = spilt_class(time_dict)
    # # # for k,v in result_dict.items():
    # # #     print(k,v)
    # # # print(new_dict,result_dict)
    # with open('time_dict_Random+LLM_106.txt', 'w') as file:
    #     json.dump(time_dict_2, file, indent=4)
    # print(len(new_dict))
    # print(result_dict)
    # for k,v in result_dict.items():
    #     print(k,v)

    # set1 = set(result_dict_1['succeed'])
    # set2 = set(result_dict_2['succeed'])
    # unique_to_set1 = set1 - set2
    # print(unique_to_set1)
    # resolve_dataset()


    # solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_smt_comp.txt'
    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1118_pre_llm_llama3.1:70b_1200s_QF_LIA.txt'
    # test_group_2_no_save(solve_name,info_name)

    #使用同一个rl agent
    # solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_bingxing.txt'
    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1111_pre_SMTimer_save_docker_llama_3.1:70b_1200s_info_dict_rl.txt'
    # test_group_2_save_1208(solve_name,info_name)
    # 只使用llm的结果 LLM
    # solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_bingxing.txt'
    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1210_pre_SMTimer_llama3.1:70b_1200s_info_dict_rl_llm_only.txt'
    # result_dict, time_dict= test_group_2_no_save_1207_only_llm(solve_name,info_name)
    # with open('time_dict_LLM_106.txt', 'w') as file:
    #     json.dump(time_dict, file, indent=4)

    # 全部随机的结果 Random+Random
    # solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_bingxing.txt'
    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1217_pre_SMTimer_llama3.1:70b_1200s_info_dict_all_random.txt'
    # # result_dict_1 = test_group_2_no_save(solve_name,info_name)
    # result_dict_2,time_dict,time_dict_2,info_dict =test_group_2_no_save_1207(solve_name, info_name)
    # # print(time_dict_2)
    # with open('time_dict_Random+Random_106.txt', 'w') as file:
    #     json.dump(time_dict_2, file, indent=4)

    # 全部随机的结果 RL+Random
    # solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_bingxing.txt'
    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1223_pre_SMTimer_llama3.1:70b_1200s_info_dict_rl_random_1223.txt'
    # # result_dict_1 = test_group_2_no_save(solve_name,info_name)
    # result_dict_2,time_dict,time_dict_2,info_dict =test_group_2_no_save_1207(solve_name, info_name)
    # # print(time_dict_2)
    # with open('time_dict_RL+Random_106.txt', 'w') as file:
    #     json.dump(time_dict_2, file, indent=4)

    # resolve_dataset_by_var_count()

    #处理数据库
    # resolve_dataset_300s()

