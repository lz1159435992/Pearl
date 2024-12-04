
import json

import os


from test_rl.test_script.utils import load_dictionary




def test_group():
    info_name = 'info_dict.txt'
    if not os.path.exists(info_name):
        # 文件不存在时，创建文件
        info_dict = {}
        with open(info_name, 'w') as file:
            json.dump(info_dict, file, indent=4)

        print(f'文件{info_name} 已创建。')
    else:
        info_dict = load_dictionary(info_name)
        print(f'文件已存在。')
    with open('/home/lz/PycharmProjects/Pearl/test_rl/result_dict.txt', 'r') as file:
        result_dict = json.load(file)

    for key, value in result_dict.items():
        python_list = json.loads(value)
        print(python_list[0])
        if python_list[0] == 'sat':
            print('sat')
        if python_list[0] == 'unsat':
            print('unsat')



if __name__ == '__main__':
    test_group()
