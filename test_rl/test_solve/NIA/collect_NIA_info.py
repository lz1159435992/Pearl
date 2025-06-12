import os
import json
import re
def merge_json_files():
    # 定义目标路径
    target_path = '/home/lz/new_disk/NIA'

    # 初始化合并后的字典
    merged_dict = {}

    # 遍历目标路径下的所有文件
    for root, dirs, files in os.walk(target_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    # 将内容合并到字典中
                    merged_dict.update(content)
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")

    # 保存合并后的字典
    print(len(merged_dict))
    output_file_name = 'NIA.json'
    with open(output_file_name, 'w', encoding='utf-8') as f:
        json.dump(merged_dict, f, ensure_ascii=False, indent=4)

    print(f"文件 {output_file_name} 已成功保存。")
def get_info():
    output_file_name = 'NIA.json'
    with open(output_file_name, 'r', encoding='utf-8') as f:
        solve_dict = json.load(f)
        # print(content)
        # 统计每个文件的行数
        for k, v in solve_dict.items():
            if v[0] == 'sat' and v[1] > 1200:
                solve_dict[k][0] = 'unknown'
                solve_dict[k][1] = 1200
            if v[0] == 'unknown' and v[1] > 1200:
                # solve_dict[k][0] = 'unknown'
                solve_dict[k][1] = 1200
    count = 0
    for k,v in solve_dict.items():
        if v[0] == 'sat' and v[1] > 300:
           count += 1
    print(len(solve_dict),count)
def get_NIA_count():
    test_path = []
    directory = '/home/lz/Downloads/non-incremental_Hierarchy/non-incremental'
    test_path.append(directory)
    # directory = '/home/lz/Downloads/incremental_Hierarchy/incremental'
    # test_path.append(directory)
    search_list = [
        # 'QF_IDL',
        # 'QF_LIA',
        # 'QF_LRA',
        'QF_NIA',
        # 'QF_NRA',
        # 'QF_RDL',
        # 'QF_UFIDL',
        # 'QF_UFLIA',
        # 'QF_UFLRA',
        # 'QF_UFNRA',
        # 'UFLRA',
        # 'UFNIA',
    ]
    logic_systems = [
        # "QF_BOOL",
        "QF_IDL", "QF_LIA", "QF_LRA", "QF_RDL",
        "QF_UF", "QF_UFIDL",
        # "QF_UFLIA", "QF_UFLRA", "QF_UFLIRA",
        # "BOOL",
        "LRA", "LIA",
        # "UFLIRA", "UFLRA",
        "QF_BV",
        "QF_UFBV",
        "QF_SLIA",
        "QF_BV", "QF_UFBV",
        # "QF_ABV", "QF_AUFBV", "QF_AUFLIA", "QF_ALIA", "QF_AX",
        # "QF_AUFBVLIRA",
        "QF_NRA", "QF_NIA",
        # "UFBV", "BV"
    ]
    # 遍历目录
    pattern = re.compile(r'\(set-info :status (\w+)\)')
    count = 0
    for directory in test_path:
        path = []
        for search in search_list:
            for dirpath, dirnames, filenames in os.walk(os.path.join(directory, search)):
                for filename in filenames:
                    # 构造完整的文件路径
                    file_path = os.path.join(dirpath, filename)
                    print(file_path)  # 或者进行其他操作
                    if 'starexec_description.txt' in file_path or '52759_b3ecd2335fd16ec2eee2_9_UFDTBV' in file_path or 'sll-optional-1.i_1' in file_path:
                        print('NOTHING ')
                    else:
                        with open(file_path, 'r') as file:
                            # 璇诲彇鏂囦欢鎵€鏈夊唴瀹瑰埌涓€涓瓧绗︿覆
                            smtlib_str = file.read()

                        match = pattern.search(smtlib_str)
                        if match:
                            status = match.group(1)
                            print(f"The status is: {status}")
                        else:
                            print("No status found in the file.")
                            return None
                        if status != 'unsat':
                            count += 1
    print(count)
if __name__ == '__main__':
    merge_json_files()
    # get_info()
    # get_NIA_count()