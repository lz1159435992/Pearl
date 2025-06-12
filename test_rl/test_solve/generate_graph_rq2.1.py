import seaborn as sns
sns.set_palette("Set2")
import matplotlib.pyplot as plt
import numpy as np
from test_rl.test_script.sql_test import load_dictionary
def deal_time_dict(file_path):
    x_list = []
    y_sat_list = []
    y_list = []
    time_dict = load_dictionary(file_path)
    for k, v in time_dict.items():
        print(k)
        for k1, v1 in v.items():
            print(k1, v1)
    buzybox_data = time_dict['buzybox_angr.tar.gz']
    # 准备数据
    buzybox_projects = list(buzybox_data.keys())
    buzybox_sat_time_avg = [buzybox_data[proj]['sat_time_avg'] for proj in buzybox_projects]
    buzybox_sat_unknown_time = [buzybox_data[proj]['sat+unknown_time_avg'] for proj in buzybox_projects]
    buzybox_avg_time = [buzybox_data[proj]['avg'] for proj in buzybox_projects]

    # 提取 gnu_angr.tar.gz 中的数据
    gnu_data = time_dict['gnu_angr.tar.gz']

    # 准备数据
    gnu_projects = list(gnu_data.keys())
    gnu_sat_time_avg = [gnu_data[proj]['sat_time_avg'] for proj in gnu_projects]
    gnu_sat_unknown_time = [gnu_data[proj]['sat+unknown_time_avg'] for proj in gnu_projects]
    gnu_avg_time = [gnu_data[proj]['avg'] for proj in gnu_projects]
    return buzybox_projects,buzybox_sat_time_avg,buzybox_sat_unknown_time,buzybox_avg_time,gnu_projects,gnu_sat_time_avg,gnu_sat_unknown_time,gnu_avg_time
buzybox_projects,buzybox_sat_time_avg,buzybox_sat_unknown_time,buzybox_avg_time,gnu_projects,gnu_sat_time_avg,gnu_sat_unknown_time,gnu_avg_time = deal_time_dict('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/result_dict_z3solver_300s.txt')
RL_LLM_buzybox_projects,RL_LLM_buzybox_sat_time_avg,RL_LLM_buzybox_sat_unknown_time,RL_LLM_buzybox_avg_time,RL_LLM_gnu_projects,RL_LLM_gnu_sat_time_avg,RL_LLM_gnu_sat_unknown_time,RL_LLM_gnu_avg_time = deal_time_dict('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/result_dict_RL+LLM_108.txt')



indices_to_remove = [i for i, x in enumerate(sat_time_avg) if x == 0]

# 使用列表推导式删除所有列表中对应位置的元素
list1 = [x for i, x in enumerate(list1) if i not in indices_to_remove]
# 设置Seaborn配色方案

# 绘制图形buzybox
plt.figure(figsize=(12, 8))


# plt.figure(figsize=(12, 6))
plt.plot(buzybox_projects, buzybox_sat_time_avg, marker='o', markersize=5,linestyle='-', label=f'buzybox_sat_time_avg')
plt.plot(buzybox_projects, RL_LLM_buzybox_sat_time_avg, marker='s', markersize=5,linestyle='-', label=f'RL+LLM_buzybox_sat_time_avg')
# plt.plot(buzybox_projects, buzybox_sat_unknown_time, marker='s', markersize=2,linestyle='--', label=f'buzybox_sat_unknown_time')
# plt.plot(buzybox_projects, buzybox_avg_time, marker='^', markersize=2,linestyle='-.', label=f'buzbox_avg_time')
# plt.plot(x_z3, y_Random_Random, marker='p', markersize=2,linestyle='-.', label=f'Random+Random (Area: {area_Random_Random:.2f})')
# plt.plot(x_z3, y_Random_LLM, marker='*', markersize=2,linestyle='-.', label=f'Random+LLM (Area: {area_Random_LLM:.2f})')
# 添加标题和坐标轴标签
plt.title('XXX')
plt.xlabel('x')
plt.ylabel('y')
# 显示图例
plt.legend(loc='best', fontsize=12)
# 显示网格
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# 调整布局
plt.tight_layout()
# 显示图形
plt.show()

# 绘制图形buzybox
plt.figure(figsize=(12, 8))


# plt.figure(figsize=(12, 6))
plt.plot(gnu_projects, gnu_sat_time_avg, marker='o', markersize=5,linestyle='-', label=f'gnu_sat_time_avg')
plt.plot(gnu_projects, RL_LLM_gnu_sat_time_avg, marker='s', markersize=5,linestyle='-', label=f'RL+LLM_gnu_sat_time_avg')
# plt.plot(buzybox_projects, buzybox_avg_time, marker='^', markersize=2,linestyle='-.', label=f'buzbox_avg_time')
# plt.plot(x_z3, y_Random_Random, marker='p', markersize=2,linestyle='-.', label=f'Random+Random (Area: {area_Random_Random:.2f})')
# plt.plot(x_z3, y_Random_LLM, marker='*', markersize=2,linestyle='-.', label=f'Random+LLM (Area: {area_Random_LLM:.2f})')
# 添加标题和坐标轴标签
plt.title('XXX')
plt.xlabel('x')
plt.ylabel('y')
# 显示图例
plt.legend(loc='best', fontsize=12)
# 显示网格
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# 调整布局
plt.tight_layout()
# 显示图形
plt.show()
