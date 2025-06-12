import time

import matplotlib.pyplot as plt
from test_rl.test_script.utils import load_dictionary
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.faker import Faker
from pyecharts_snapshot.main import make_a_snapshot

# 更新后的函数来绘制图表并保存为 PDF，使用seaborn的'muted'调色板
def plot_pyecharts(data, tool_name,path):
    apps = list(data.keys())
    sat_counts = [data[app]['sat'] for app in apps]
    # unsat_counts = [data[app]['unsat'] for app in apps]
    unknown_counts = [data[app]['unknown'] for app in apps]
    sat_times = [data[app]['sat_time'] for app in apps]
    # unsat_times = [data[app]['unsat_time'] for app in apps]
    unknown_times = [data[app]['unknown_time'] for app in apps]
    sat_avg_times = [data[app]['sat_time_avg'] for app in apps]
    # unsat_avg_times = [data[app]['unsat_time_avg'] for app in apps]
    unknown_avg_times = [data[app]['unknown_time_avg'] for app in apps]


    c = (
        Bar()
        .add_xaxis(apps)
        .add_yaxis("sat_counts", sat_counts, stack="stack1")
        .add_yaxis("unknown_counts", unknown_counts, stack="stack1")
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(title_opts=opts.TitleOpts(title="Bar-堆叠数据（全部）"))
        .render(f'{path}/{tool_name}_counts.html')
    )
    time.sleep(5)
    make_a_snapshot(f'{path}/{tool_name}_counts.html', f'{path}/{tool_name}_counts.png')


def plot_solver_data_to_pdf(data, tool_name,path):
    apps = list(data.keys())
    sat_counts = [data[app]['sat'] for app in apps]
    # unsat_counts = [data[app]['unsat'] for app in apps]
    unknown_counts = [data[app]['unknown'] for app in apps]
    sat_times = [data[app]['sat_time'] for app in apps]
    # unsat_times = [data[app]['unsat_time'] for app in apps]
    unknown_times = [data[app]['unknown_time'] for app in apps]
    sat_avg_times = [data[app]['sat_time_avg'] for app in apps]
    # unsat_avg_times = [data[app]['unsat_time_avg'] for app in apps]
    unknown_avg_times = [data[app]['unknown_time_avg'] for app in apps]

    # 使用seaborn的'muted'调色板
    sns.set_palette('deep')

    # 绘制数量图并保存为 PDF
    plt.figure(figsize=(10, 6))
    plt.bar(apps, sat_counts, alpha=0.7, label='SAT')
    # plt.bar(apps, unsat_counts, alpha=0.7, label='UNSAT')
    plt.bar(apps, unknown_counts, alpha=0.7, label='UNKNOWN')
    plt.xlabel('Applications')
    plt.ylabel('Counts')
    plt.title(f'{tool_name} - SAT, UNSAT, UNKNOWN Counts')
    plt.legend()
    plt.xticks(rotation=90)
    sanitized_tool_name = tool_name
    pdf_path_counts = f'{path}/{sanitized_tool_name}_counts.pdf'
    with PdfPages(pdf_path_counts) as pdf:
        pdf.savefig()
    plt.close()

    # 绘制时间图并保存为 PDF
    plt.figure(figsize=(10, 6))
    plt.bar(apps, sat_times, alpha=0.7, label='SAT Time')
    # plt.bar(apps, unsat_times, alpha=0.7, label='UNSAT Time')
    plt.bar(apps, unknown_times, alpha=0.7, label='UNKNOWN Time')
    plt.xlabel('Applications')
    plt.ylabel('Times')
    plt.title(f'{tool_name} - SAT, UNSAT, UNKNOWN Times')
    plt.legend()
    plt.xticks(rotation=90)
    pdf_path_times = f'{path}/{sanitized_tool_name}_times.pdf'
    with PdfPages(pdf_path_times) as pdf:
        pdf.savefig()
    plt.close()

    # 绘制平均时间图并保存为 PDF
    plt.figure(figsize=(10, 6))
    plt.bar(apps, sat_avg_times, alpha=0.7, label='Average SAT Time')
    # plt.bar(apps, unsat_avg_times, alpha=0.7, label='Average UNSAT Time')
    plt.bar(apps, unknown_avg_times, alpha=0.7, label='Average UNKNOWN Time')
    plt.xlabel('Applications')
    plt.ylabel('Average Times')
    plt.title(f'{tool_name} - Average SAT, UNSAT, UNKNOWN Times')
    plt.legend()
    plt.xticks(rotation=90)
    pdf_path_avg_times = f'{path}/{sanitized_tool_name}_avg_times.pdf'
    with PdfPages(pdf_path_avg_times) as pdf:
        pdf.savefig()
    plt.close()

    return pdf_path_counts, pdf_path_times, pdf_path_avg_times

# 为每个工具生成 PDF 文件
# 解析数据字符串为字典
data_dict = load_dictionary('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/result_dict_z3solver_300s.txt')

# # 调用函数为每个工具生成 PDF 文件
# plot_solver_data_to_pdf(data_dict['buzybox_angr.tar.gz'], 'buzybox','Z3solver')
# plot_solver_data_to_pdf(data_dict['gnu_angr.tar.gz'], 'gnu','Z3solver')
plot_pyecharts(data_dict['buzybox_angr.tar.gz'], 'buzybox','Z3solver')