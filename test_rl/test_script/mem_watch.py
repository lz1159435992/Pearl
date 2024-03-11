import psutil

import GPUtil

import time

import datetime


def log_system_usage():
    # 定义文件名，以当前日期和时间命名

    file_name = f"system_usage_log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

    with open(file_name, 'w') as log_file:

        while True:

            # 获取系统内存使用情况

            memory = psutil.virtual_memory()

            # 获取所有GPU的信息

            gpus = GPUtil.getGPUs()

            # 记录当前时间

            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            log_file.write(f"{current_time} - System Memory Usage: {memory.percent}%\n")

            for gpu in gpus:
                log_file.write(
                    f"{current_time} - GPU {gpu.id}: {gpu.name}, Utilization: {gpu.load * 100}%, Memory Usage: {gpu.memoryUsed}/{gpu.memoryTotal} MB\n")

            log_file.flush()  # 确保写入磁盘

            time.sleep(5)  # 每5秒钟检测一次


# 运行函数

log_system_usage()