import numpy as np
import matplotlib.pyplot as plt
import os
from test_rl.test_script.sql_test import load_dictionary

# 设置数据文件路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 加载数据
time_dict_z3 = load_dictionary(os.path.join(current_dir, 'time_dict_z3solver_106.txt'))
time_dict_RL_LLM = load_dictionary(os.path.join(current_dir, 'time_dict_RL+LLM_106.txt'))
time_dict_LLM = load_dictionary(os.path.join(current_dir, 'time_dict_LLM_106.txt'))
time_dict_Random_Random = load_dictionary(os.path.join(current_dir, 'time_dict_Random+Random_106.txt'))
time_dict_Random_LLM = load_dictionary(os.path.join(current_dir, 'time_dict_Random+LLM_106.txt'))
time_dict_RL_Random = load_dictionary(os.path.join(current_dir, 'time_dict_RL+Random_106.txt'))

# 数据预处理函数
def process_time_data(time_dict):
    """处理时间数据，只保留成功求解的案例并排序"""
    success_dict = {k: v for k, v in time_dict.items() if v > 0}
    return dict(sorted(success_dict.items(), key=lambda item: item[1]))

# 处理所有数据
sorted_dict_z3 = process_time_data(time_dict_z3)
sorted_dict_RL_LLM = process_time_data(time_dict_RL_LLM)
sorted_dict_LLM = process_time_data(time_dict_LLM)
sorted_dict_Random_Random = process_time_data(time_dict_Random_Random)
sorted_dict_Random_LLM = process_time_data(time_dict_Random_LLM)
sorted_dict_RL_Random = process_time_data(time_dict_RL_Random)

# 设置图表样式
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 300

# 创建图表
fig, ax = plt.subplots()

# 准备数据
data_sets = {
    'Random+Random': sorted_dict_Random_Random,
    'LLM': sorted_dict_LLM,
    'Random+LLM': sorted_dict_Random_LLM,
    'RL+Random': sorted_dict_RL_Random,
    'RL+LLM': sorted_dict_RL_LLM
}

# 设置线条样式
styles = {
    'Random+Random': {'marker': 'h', 'linestyle': '--', 'markersize': 4, 'color': '#66c2a5'},
    'LLM': {'marker': 'D', 'linestyle': ':', 'markersize': 4, 'color': '#fc8d62'},
    'Random+LLM': {'marker': 'H', 'linestyle': '-.', 'markersize': 4, 'color': '#8da0cb'},
    'RL+Random': {'marker': '8', 'linestyle': ':', 'markersize': 4, 'color': '#e78ac3'},
    'RL+LLM': {'marker': 'v', 'linestyle': '-', 'markersize': 4, 'color': '#a6d854'}
}

# 绘制曲线
for name, data in data_sets.items():
    y = list(data.values())
    x = range(len(y))
    plt.plot(x, y, label=f'{name} ({len(y)} solved)', **styles[name], linewidth=1.5)

# 设置坐标轴
plt.xlabel('Solved Constraints Instance Index')
plt.ylabel('Solving Time (s)')
plt.title('Solving Time Distribution (Successful Cases Only)')

# 添加网格
plt.grid(True, linestyle='--', alpha=0.3)

# 设置图例
plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=10)

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('solving_time_comparison_success.pdf', bbox_inches='tight', format='pdf')
plt.savefig('solving_time_comparison_success.png', bbox_inches='tight', dpi=300)

# 显示图表
plt.show()

# 计算统计信息
def calculate_statistics(time_dict):
    """计算求解时间的统计信息"""
    times = np.array(list(time_dict.values()))
    success_times = times[times > 0]
    timeout_times = times[times <= 0]
    
    return {
        'total_instances': len(times),
        'success_count': len(success_times),
        'timeout_count': len(timeout_times),
        'success_rate': len(success_times) / len(times) * 100,
        'avg_success_time': np.mean(success_times) if len(success_times) > 0 else 0,
        'min_success_time': np.min(success_times) if len(success_times) > 0 else 0,
        'max_success_time': np.max(success_times) if len(success_times) > 0 else 0
    }

# 打印统计信息
print("\n求解性能统计:")
print("-" * 50)
for name, data in data_sets.items():
    stats = calculate_statistics(time_dict_z3 if name == 'z3' else data)
    print(f"\n{name}:")
    print(f"总实例数: {stats['total_instances']}")
    print(f"成功求解数: {stats['success_count']}")
    print(f"成功率: {stats['success_rate']:.2f}%")
    print(f"平均求解时间: {stats['avg_success_time']:.2f}s")
    print(f"最短求解时间: {stats['min_success_time']:.2f}s")
    print(f"最长求解时间: {stats['max_success_time']:.2f}s") 