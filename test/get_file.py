import os
directory = '/home/nju/Downloads/QF_FP'
for dirpath, dirnames, filenames in os.walk(directory):
    for filename in filenames:
        # 构造完整的文件路径
        file_path = os.path.join(dirpath, filename)
        print(file_path)  # 或者进行其他操作