import matplotlib.pyplot as plt
from test_rl.test_script.utils import load_dictionary
from matplotlib.backends.backend_pdf import PdfPages

# 解析数据字符串为字典
data_dict = load_dictionary('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/result_dict_z3solver.txt')


# 定义一个函数来绘制图表并保存为 PDF
def plot_solver_data_to_pdf(data, tool_name):
    # 创建包含应用程序和其对应数据的元组列表
    app_sat_counts = [(k, v['sat']) for k, v in data.items()]
    app_unsat_counts = [(k, v['unsat']) for k, v in data.items()]
    app_unknown_counts = [(k, v['unknown']) for k, v in data.items()]
    app_sat_times = [(k, v['sat_time']) for k, v in data.items()]
    app_unsat_times = [(k, v['unsat_time']) for k, v in data.items()]
    app_unknown_times = [(k, v['unknown_time']) for k, v in data.items()]
    app_sat_avg_times = [(k, v['sat_time_avg']) for k, v in data.items()]
    app_unsat_avg_times = [(k, v['unsat_time_avg']) for k, v in data.items()]
    app_unknown_avg_times = [(k, v['unknown_time_avg']) for k, v in data.items()]

    # 根据数据值对应用程序的顺序进行排序
    apps_sorted_by_sat_counts = sorted(app_sat_counts, key=lambda x: x[1])
    apps_sorted_by_unsat_counts = sorted(app_unsat_counts, key=lambda x: x[1])
    apps_sorted_by_unknown_counts = sorted(app_unknown_counts, key=lambda x: x[1])
    apps_sorted_by_sat_times = sorted(app_sat_times, key=lambda x: x[1])
    apps_sorted_by_unsat_times = sorted(app_unsat_times, key=lambda x: x[1])
    apps_sorted_by_unknown_times = sorted(app_unknown_times, key=lambda x: x[1])
    apps_sorted_by_sat_avg_times = sorted(app_sat_avg_times, key=lambda x: x[1])
    apps_sorted_by_unsat_avg_times = sorted(app_unsat_avg_times, key=lambda x: x[1])
    apps_sorted_by_unknown_avg_times = sorted(app_unknown_avg_times, key=lambda x: x[1])

    # 提取排序后的应用程序名称和对应的数据
    apps = [app[0] for app in apps_sorted_by_sat_counts]
    sat_counts = [app[1] for app in apps_sorted_by_sat_counts]
    unsat_counts = [app[1] for app in apps_sorted_by_unsat_counts]
    unknown_counts = [app[1] for app in apps_sorted_by_unknown_counts]
    sat_times = [app[1] for app in apps_sorted_by_sat_times]
    unsat_times = [app[1] for app in apps_sorted_by_unsat_times]
    unknown_times = [app[1] for app in apps_sorted_by_unknown_times]
    sat_avg_times = [app[1] for app in apps_sorted_by_sat_avg_times]
    unsat_avg_times = [app[1] for app in apps_sorted_by_unsat_avg_times]
    unknown_avg_times = [app[1] for app in apps_sorted_by_unknown_avg_times]

    # 定义学术风格的颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # 创建 PDF 文件
    # 绘制数量图并保存为 PDF
    plt.figure(figsize=(10, 6))
    plt.bar(apps, sat_counts, color=colors[0], alpha=0.7, label='SAT')
    plt.bar(apps, unsat_counts, color=colors[1], alpha=0.7, label='UNSAT')
    plt.bar(apps, unknown_counts, color=colors[2], alpha=0.7, label='UNKNOWN')
    plt.yscale('log')
    plt.xlabel('Applications')
    plt.ylabel('Counts (log scale)')
    plt.title(f'{tool_name} - SAT, UNSAT, UNKNOWN Counts')
    plt.legend()
    plt.xticks(rotation=90)
    with PdfPages(f'{tool_name}_counts.pdf') as pdf:
        pdf.savefig()
    plt.close()

    # 绘制时间图并保存为 PDF
    plt.figure(figsize=(10, 6))
    plt.bar(apps, sat_times, color=colors[0], alpha=0.7, label='SAT Time')
    plt.bar(apps, unsat_times, color=colors[1], alpha=0.7, label='UNSAT Time')
    plt.bar(apps, unknown_times, color=colors[2], alpha=0.7, label='UNKNOWN Time')
    plt.yscale('log')
    plt.xlabel('Applications')
    plt.ylabel('Times (log scale)')
    plt.title(f'{tool_name} - SAT, UNSAT, UNKNOWN Times')
    plt.legend()
    plt.xticks(rotation=90)
    with PdfPages(f'{tool_name}_times.pdf') as pdf:
        pdf.savefig()
    plt.close()

    # 绘制平均时间图并保存为 PDF
    plt.figure(figsize=(10, 6))
    plt.bar(apps, sat_avg_times, color=colors[0], alpha=0.7, label='Average SAT Time')
    plt.bar(apps, unsat_avg_times, color=colors[1], alpha=0.7, label='Average UNSAT Time')
    plt.bar(apps, unknown_avg_times, color=colors[2], alpha=0.7, label='Average UNKNOWN Time')
    plt.yscale('log')
    plt.xlabel('Applications')
    plt.ylabel('Average Times (log scale)')
    plt.title(f'{tool_name} - Average SAT, UNSAT, UNKNOWN Times')
    plt.legend()
    plt.xticks(rotation=90)
    with PdfPages(f'{tool_name}_avg_times.pdf') as pdf:
        pdf.savefig()
    plt.close()


# 调用函数为每个工具生成 PDF 文件
plot_solver_data_to_pdf(data_dict['buzybox_angr.tar.gz'], 'buzybox')
plot_solver_data_to_pdf(data_dict['gnu_angr.tar.gz'], 'gnu')