import json

import tqdm

from test_rl.test_script.utils import normalize_smt_str


def test_group_bert_normalize_1by1_smt():
    with open('smt_v2.json', 'r') as file:
        result_dict = json.load(file)
    features_list = []
    # labels_list = []
    time_list = []
    # embedder = CodeEmbedder_normalize()
    logic_systems = [
        "QF_BOOL", "QF_IDL", "QF_LIA", "QF_LRA", "QF_RDL", "QF_UF", "QF_UFIDL",
        "QF_UFLIA", "QF_UFLRA", "QF_UFLIRA",
        "BOOL", "LRA", "LIA", "UFLIRA", "UFLRA",
        "QF_BV", "QF_UFBV",
        "QF_SLIA",
        "QF_BV", "QF_UFBV",
        "QF_ABV", "QF_AUFBV", "QF_AUFLIA", "QF_ALIA", "QF_AX",
        "QF_AUFBVLIRA",
        "QF_NRA", "QF_NIA", "UFBV", "BV"
    ]
    # 遍历字典并统计数据
    for (i, (file_path,v)) in enumerate(tqdm.tqdm(result_dict.items())):
        # 读取文件内容
        # print(file_path.split('/')[6])
        if len(v) == 0 and file_path.split('/')[6] not in logic_systems:
            continue
        with open(file_path, 'r') as file:
            smtlib_str = file.read()
            file.close()

        # 处理smtlib_str

        smtlib_str, var_dict,_ = normalize_smt_str(smtlib_str)
if __name__ == '__main__':
    test_group_bert_normalize_1by1_smt()