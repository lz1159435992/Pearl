import matplotlib.pyplot as plt

from test_rl.test_script.utils import load_dictionary

# 解析数据字符串为字典
data_dict = load_dictionary('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/result_dict_z3solver_300s.txt')

# 定义一个函数来绘制图表并保存为 PDF
def plot_solver_data_to_pdf(data, tool_name):
    apps = list(data.keys())
    sat_counts = [data[app]['sat'] for app in apps]
    unsat_counts = [data[app]['unsat'] for app in apps]
    unknown_counts = [data[app]['unknown'] for app in apps]
    sat_times = [data[app]['sat_time'] for app in apps]
    unsat_times = [data[app]['unsat_time'] for app in apps]
    unknown_times = [data[app]['unknown_time'] for app in apps]
    sat_avg_times = [data[app]['sat_time_avg'] for app in apps]
    unsat_avg_times = [data[app]['unsat_time_avg'] for app in apps]
    unknown_avg_times = [data[app]['unknown_time_avg'] for app in apps]

    # 定义学术风格的颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # 创建 PDF 文件
    from matplotlib.backends.backend_pdf import PdfPages
    # 绘制数量图并保存为 PDF
    plt.figure(figsize=(10, 6))
    plt.bar(apps, sat_counts, alpha=0.7, label='SAT')
    # plt.bar(apps, unsat_counts, color=colors[1], alpha=0.7, label='UNSAT')
    plt.bar(apps, unknown_counts, alpha=0.7, label='UNKNOWN')
    # plt.yscale('log')
    plt.xlabel('Applications')
    plt.ylabel('Counts (log scale)')
    plt.title(f'{tool_name} - SAT, UNSAT, UNKNOWN Counts')
    plt.legend()
    plt.xticks(rotation=90)

    with PdfPages(f'{tool_name}_counts.pdf') as pdf:
        pdf.savefig()
    plt.show()
    plt.close()

    # 绘制时间图并保存为 PDF
    plt.figure(figsize=(10, 6))
    plt.bar(apps, sat_times, alpha=0.7, label='SAT Time')
    # plt.bar(apps, unsat_times, color=colors[1], alpha=0.7, label='UNSAT Time')
    plt.bar(apps, unknown_times, alpha=0.7, label='UNKNOWN Time')
    # plt.yscale('log')
    plt.xlabel('Applications')
    plt.ylabel('Times (log scale)')
    plt.title(f'{tool_name} - SAT, UNSAT, UNKNOWN Times')
    plt.legend()
    plt.xticks(rotation=90)

    with PdfPages(f'{tool_name}_times.pdf') as pdf:
        pdf.savefig()
    plt.show()
    plt.close()
    # 绘制平均时间图并保存为 PDF
    plt.figure(figsize=(10, 6))
    # print(sat_avg_times)
    plt.bar(apps, sat_avg_times, alpha=0.7, label='Average SAT Time')
    # plt.bar(apps, unsat_avg_times, color=colors[1], alpha=0.7, label='Average UNSAT Time')
    plt.bar(apps, unknown_avg_times, alpha=0.7, label='Average UNKNOWN Time')
    # plt.yscale('log')
    plt.xlabel('Applications')
    plt.ylabel('Average Times (log scale)')
    plt.title(f'{tool_name} - Average SAT, UNSAT, UNKNOWN Times')
    plt.legend()
    plt.xticks(rotation=90)

    with PdfPages(f'{tool_name}_avg_times.pdf') as pdf:
        pdf.savefig()
    plt.show()
    plt.close()

# 调用函数为每个工具生成 PDF 文件
plot_solver_data_to_pdf(data_dict['buzybox_angr.tar.gz'], 'Z3solver/buzybox')
plot_solver_data_to_pdf(data_dict['gnu_angr.tar.gz'], 'Z3solver/gnu')
