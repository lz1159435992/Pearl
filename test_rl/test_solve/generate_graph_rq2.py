
import numpy as np
from test_rl.test_script.sql_test import load_dictionary

time_dict_z3 = load_dictionary('time_dict_z3solver_106.txt')
time_dict_RL_LLM = load_dictionary('time_dict_RL+LLM_106.txt')
time_dict_LLM = load_dictionary('time_dict_LLM_106.txt')
time_dict_Random_Random = load_dictionary('time_dict_Random+Random_106.txt')
time_dict_Random_LLM = load_dictionary('time_dict_Random+LLM_106.txt')
time_dict_RL_Random = load_dictionary('time_dict_RL+Random_106.txt')

sorted_dict_z3 = dict(sorted(time_dict_z3.items(), key=lambda item: item[1]))
sorted_dict_RL_LLM = dict(sorted(time_dict_RL_LLM.items(), key=lambda item: item[1]))
sorted_dict_LLM = dict(sorted(time_dict_LLM.items(), key=lambda item: item[1]))
sorted_dict_Random_Random = dict(sorted(time_dict_Random_Random.items(), key=lambda item: item[1]))
sorted_dict_Random_LLM = dict(sorted(time_dict_Random_LLM.items(), key=lambda item: item[1]))
sorted_dict_RL_Random = dict(sorted(time_dict_RL_Random.items(), key=lambda item: item[1]))
import seaborn as sns

# 设置Seaborn配色方案
sns.set_palette("Set2")
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

y_z3 = list(sorted_dict_z3.values())
x_z3 = range(len(y_z3))
y_RL_LLM = list(sorted_dict_RL_LLM.values())
y_LLM = list(sorted_dict_LLM.values())
y_Random_Random = list(sorted_dict_Random_Random.values())
y_Random_LLM = list(sorted_dict_Random_LLM.values())
y_RL_Random = list(sorted_dict_RL_Random.values())
# 假设x_z3和y_z3等是你的数据点
x_z3 = np.array(x_z3)
y_z3 = np.array(y_z3)
y_LLM = np.array(y_LLM)
y_RL_LLM = np.array(y_RL_LLM)
y_Random_Random = np.array(y_Random_Random)
y_Random_LLM = np.array(y_Random_LLM)
y_RL_Random = np.array(y_RL_Random)
# 计算每条曲线下的面积
# area_z3 = np.trapz(y_z3, x_z3)
# area_LLM = np.trapz(y_LLM, x_z3)
# area_RL_LLM = np.trapz(y_RL_LLM, x_z3)
# area_Random_Random = np.trapz(y_Random_Random, x_z3)
# area_Random_LLM = np.trapz(y_Random_LLM, x_z3)

# 绘制图形
# plt.figure(figsize=(12, 6))

# plt.plot(x_z3, y_z3, marker='x', markersize=3, linestyle='-', label=f'z3 solver')
plt.plot(x_z3, y_Random_Random, marker='h', markersize=3, linestyle='--', label=f'Random+Random')
plt.plot(x_z3, y_LLM, marker='D', markersize=3, linestyle=':', label=f'LLM')
plt.plot(x_z3, y_Random_LLM, marker='H', markersize=3, linestyle='-.', label=f'Random+LLM')

plt.plot(x_z3, y_RL_Random, marker='8', markersize=3, linestyle=':', label=f'RL+Random')
plt.plot(x_z3, y_RL_LLM, marker='v', markersize=3, linestyle='-', label=f'RL+LLM')



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
time_sat = 0
time_unsat = 0
for k,v in sorted_dict_z3.items():
    print(k,v)
    if v>0:
        time_sat+=v
    else:
        time_unsat+=v
print(time_sat,time_unsat)