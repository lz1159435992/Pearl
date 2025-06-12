import time

import matplotlib.pyplot as plt
from test_rl.test_script.utils import load_dictionary
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from pyecharts import options as opts
from pyecharts.charts import Bar, Grid
from pyecharts.faker import Faker
from pyecharts_snapshot.main import make_a_snapshot
colors = ["#3988c5","#abd0eb"]

import time
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts_snapshot.main import make_a_snapshot

colors = ["#3988c5", "#abd0eb"]


def sort_bar_chart_data(data_dict):
    """
    对柱状图数据进行排序处理
    :param data_dict: 包含柱状图数据的字典
    :return: 排序后的数据字典
    """
    # 提取第一个工具的数据
    data1 = data_dict['buzybox_angr.tar.gz']
    apps_1 = list(data1.keys())
    sat_counts_1 = [data1[app]['sat'] for app in apps_1]
    unknown_counts_1 = [data1[app]['unknown'] for app in apps_1]

    # 提取第二个工具的数据
    data2 = data_dict['gnu_angr.tar.gz']
    apps_2 = list(data2.keys())
    sat_counts_2 = [data2[app]['sat'] for app in apps_2]
    unknown_counts_2 = [data2[app]['unknown'] for app in apps_2]

    # 合并数据
    apps = apps_1 + apps_2
    sat_counts = sat_counts_1 + sat_counts_2
    unknown_counts = unknown_counts_1 + unknown_counts_2

    # 计算总高度（成功+失败）
    total_heights = [sat + unknown for sat, unknown in zip(sat_counts, unknown_counts)]

    # 按总高度降序排序
    sorted_indices = sorted(range(len(total_heights)), key=lambda i: -total_heights[i])

    # 创建排序后的数据字典
    sorted_data = {
        'apps': [apps[i] for i in sorted_indices],
        'sat_counts': [sat_counts[i] for i in sorted_indices],
        'unknown_counts': [unknown_counts[i] for i in sorted_indices],
        'total_heights': [total_heights[i] for i in sorted_indices]
    }

    return sorted_data


def plot_pyecharts(data_dict, tool_name, path):
    # 处理第一个工具的数据
    data1 = data_dict['buzybox_angr.tar.gz']
    apps_1 = list(data1.keys())
    sat_counts_1 = [data1[app]['sat'] for app in apps_1]
    unknown_counts_1 = [data1[app]['unknown'] for app in apps_1]

    # 处理第二个工具的数据
    data2 = data_dict['gnu_angr.tar.gz']
    apps_2 = list(data2.keys())
    sat_counts_2 = [data2[app]['sat'] for app in apps_2]
    unknown_counts_2 = [data2[app]['unknown'] for app in apps_2]

    # 合并数据
    apps = apps_1 + apps_2
    sat_counts = sat_counts_1 + sat_counts_2
    unknown_counts = unknown_counts_1 + unknown_counts_2

    # 排序
    total_heights = [sat + unknown for sat, unknown in zip(sat_counts, unknown_counts)]
    sorted_indices = sorted(range(len(total_heights)), key=lambda i: -total_heights[i])

    sorted_data = {
        'apps': [apps[i] for i in sorted_indices],
        'sat_counts': [sat_counts[i] for i in sorted_indices],
        'unknown_counts': [unknown_counts[i] for i in sorted_indices],
        'total_heights': [total_heights[i] for i in sorted_indices]
    }

    for i in range(len(sorted_data['apps'])):
        if sorted_data['sat_counts'][i] == 0:
            sorted_data['sat_counts'][i] = ''
        if sorted_data['unknown_counts'][i] == 0:
            sorted_data['unknown_counts'][i] = ''

    # 创建柱状图
    c = (
        Bar(init_opts=opts.InitOpts(renderer='svg', width="1200px", height="600px", bg_color='white'))
        .add_xaxis(sorted_data['apps'])
        .add_yaxis(
            "succeed",
            sorted_data['sat_counts'],
            stack="stack1",
            color=colors[0],
            label_opts=opts.LabelOpts(
                is_show=True,
                font_size=24  # 将柱子上的数字字体调大（原来12左右的话，这里*2大约24）
            )
        )
        .add_yaxis(
            "failed",
            sorted_data['unknown_counts'],
            stack="stack1",
            color=colors[1],
            label_opts=opts.LabelOpts(
                is_show=True,
                font_size=24,  # 同上
                formatter=lambda params: str(params.value) if params.value != 0 else ""
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="",
                title_textstyle_opts=opts.TextStyleOpts(font_size=32)  # 标题字体变大
            ),
            legend_opts=opts.LegendOpts(
                item_width=50,  # 原来25左右，这里放大2倍
                item_height=25,  # 原来12左右，这里放大2倍
                textstyle_opts=opts.TextStyleOpts(font_size=24)  # 图例文字字体放大
            ),
            xaxis_opts=opts.AxisOpts(
                name="Applications",
                axislabel_opts=opts.LabelOpts(
                    rotate=90,
                    interval=0,
                    font_size=20  # x轴字体放大
                ),
                name_textstyle_opts=opts.TextStyleOpts(font_size=24)  # x轴标题放大
            ),
            yaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(font_size=20),  # y轴字体放大
                name_textstyle_opts=opts.TextStyleOpts(font_size=24)  # y轴标题放大
            )
        )
    )
    grid = (Grid(init_opts=opts.InitOpts(
        renderer='svg', width='1400px', height='600px', bg_color='white')
    ).add(c,
          grid_opts=opts.GridOpts(pos_right="15%",pos_bottom="15%") ) )
    # 渲染图表为HTML
    html_path = f'{path}/{tool_name}_counts.html'
    grid.render(html_path)

    # 使用自定义 JavaScript 来隐藏值为 0 的标签
    with open(html_path, 'r') as file:
        content = file.read()

    content = content.replace(
        '<span class="echarts-text-value">',
        '''
        <span class="echarts-text-value" style="display: none;">
            0
        </span>
        <span class="echarts-text-value">
        '''
    )

    with open(html_path, 'w') as file:
        file.write(content)


# 解析数据字符串为字典
data_dict = load_dictionary('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/result_dict_z3solver_300s.txt')

# 调用函数为每个工具生成 PDF 文件
plot_pyecharts(data_dict, 'buzybox', 'Z3solver')

# 解析数据字符串为字典
data_dict = load_dictionary('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/result_dict_RL+LLM_108.txt')

# 调用函数为每个工具生成 PDF 文件
plot_pyecharts(data_dict, 'buzybox', 'RL_LLM')