import pyecharts.options as opts
from matplotlib import pyplot as plt
from pyecharts.charts import Grid, Boxplot, Scatter

from test_rl.test_script.utils import load_dictionary

result_dict_Z3 = load_dictionary('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/result_dict_z3solver_300s.txt')
result_dict_RL_LLM = load_dictionary('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/result_dict_RL+LLM_108.txt')

def box_plot_2(data_dict):
    y_data = []
    # 处理第一个工具的数据
    data1 = data_dict['buzybox_angr.tar.gz']
    apps_1 = list(data1.keys())
    for k, v in data1.items():
        y_data.append(v['sat_list'])


    # 处理第二个工具的数据
    data2 = data_dict['gnu_angr.tar.gz']
    apps_2 = list(data2.keys())
    for k, v in data2.items():
        y_data.append(v['sat_list'])




    x_data = apps_1+apps_2

    x_data_numeric = []
    y_data_expanded = []

    for i, y_list in enumerate(y_data):
        x_data_numeric.extend([i] * len(y_list))
        y_data_expanded.extend(y_list)
    print(y_data)
    # 绘制散点图
    plt.scatter(x_data_numeric, y_data_expanded, s = 5)

    # 设置图表标题和标签
    plt.title('XXX')
    plt.xlabel('Applications')
    plt.ylabel('time')

    # 设置x轴的刻度标签，并旋转90度
    plt.xticks(ticks=range(len(x_data)), labels=x_data, rotation=90)

    # 显示图表
    plt.show()
box_plot_2(result_dict_Z3)
box_plot_2(result_dict_RL_LLM)