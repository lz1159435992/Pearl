import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Subset
import glob
import os
from test_rl.test_script.utils import parse_smt2_in_parts, process_smt_lib_string, fetch_data_as_dict, \
    solve_and_measure_time, model_to_dict, load_dictionary, extract_variables_from_smt2_content, normalize_variables, normalize_smt_str
from test_rl.bert_embedder_test import CodeEmbedder_normalize


class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train():
    with open('QF_IDF_train.json', 'r') as file:
        train_dict = json.load(file)

    with open('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_smt_comp.txt', 'r') as file:
        solve_dict = json.load(file)
    # with open('/home/lz/PycharmProjects/Pearl/test_rl/predictor/embeding_QF_IDL.json', 'r') as file:
    #     embed_dict = json.load(file)

    # # 定义文件名模式，这里假设文件名以 'feature_normal_' 开头并以 '.npy' 结尾
    # file_pattern = 'features_normal_*.npy'
    #
    # # 使用 glob.glob 找到所有匹配的文件，并根据文件名中的数字进行排序
    # file_paths = sorted(glob.glob(os.path.join('/home/lz/PycharmProjects/Pearl/test_rl/features', file_pattern)),
    #                     key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))

    # 初始化一个空列表来收集所有的数组
    features_list = []
    labels_list = []
    times_list = []

    for k,v in train_dict.items():
        # 逐个加载每个文件并添加到列表中
        features_list.append(np.load(v[0]))
        if solve_dict[k][0] == 'sat':
            labels_list.append(0)
        else:
            labels_list.append(1)
        times_list.append(v[1])

    labels_list = np.array(labels_list)
    times_list = np.array(times_list)

    # 使用 numpy.vstack 将所有数组垂直堆叠起来
    # 如果数组的维度相同，也可以使用 numpy.concatenate(features_list, axis=0)
    train_features = np.concatenate(features_list, axis=0)

    train_labels = labels_list
    time_labels = times_list

    # train_labels = np.concatenate(labels_list, axis=0)
    # time_labels = np.concatenate(times_list, axis=0)

    train_features = torch.tensor(train_features, dtype=torch.float32).squeeze()
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    time_labels = torch.tensor(time_labels, dtype=torch.float32)

    # features_array = np.array(features_list)
    # labels_array = np.array(labels_list)
    # time_array = np.array(times_list)
    # # 使用 numpy.vstack 将所有数组垂直堆叠起来
    # # 如果数组的维度相同，也可以使用 numpy.concatenate(features_list, axis=0)


    # train_features = np.concatenate(features_list, axis=0)

    # train_features = np.load('features.npy')
    # train_labels = np.load('labels.npy')
    # time_labels = np.load('time.npy')

    # train_features = torch.tensor(train_features, dtype=torch.float32).squeeze()
    # train_labels = torch.tensor(train_labels, dtype=torch.float32)
    # time_labels = torch.tensor(time_labels, dtype=torch.float32)

    dataset = TensorDataset(train_features, train_labels)

    indices = torch.randperm(len(dataset))
    split_idx = int(0.8 * len(dataset))

    train_dataset = Subset(dataset, indices[:split_idx])
    test_dataset = Subset(dataset, indices[split_idx:])

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleClassifier()
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
    # ...之前的代码...

    # 定义优化器和学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # 初始化最佳验证损失和早停计数
    best_val_loss = float('inf')
    no_improve_epochs = 0
    early_stopping_patience = 10

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            print(data)
            print(outputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader)}')

        # 更新学习率
        scheduler.step(epoch_loss)

        # 早停检查
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'QF_IDL_bert_predictor_mask_best.pth')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print(f'Early stopping after {no_improve_epochs} epochs without improvement.')
                break  # 停止训练循环

        # ...测试集评估代码...
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data, labels in test_loader:
                outputs = model(data)
                predicted = (outputs > 0.5).float()  # Convert logits to predictions (0 or 1)
                total += labels.size(0)
                correct += (predicted.squeeze() == labels).sum().item()

            print(f'Accuracy of the model on the test images: {100 * correct / total} %')
    # 保存最终模型
    torch.save(model.state_dict(), 'QF_IDL_bert_predictor_mask_final.pth')


def test():
    model = SimpleClassifier()

    with open('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_smt_comp.txt', 'r') as file:
        solve_dict = json.load(file)

    # 步骤3: 加载保存的状态字典
    # 假设你保存的文件名为 'bert_predictor_mask_best.pth' 或 'bert_predictor_mask_final.pth'
    # 你可以根据需要加载最佳模型或最终模型
    if not os.path.exists('QF_IDL_bert_predictor_mask_best.pth'):
        model_path = 'QF_IDL_bert_predictor_mask_final.pth'
    else:
        model_path = 'QF_IDL_bert_predictor_mask_best.pth'
    state_dict = torch.load(model_path)

    # 步骤4: 将状态字典应用到模型
    model.load_state_dict(state_dict)

    with open('QF_IDF_test.json', 'r') as file:
        test_dict = json.load(file)

    features_list = []
    labels_list = []
    times_list = []
    for k, v in test_dict.items():
        # 逐个加载每个文件并添加到列表中
        features_list.append(np.load(v[0]))
        if solve_dict[k][0] == 'sat':
            labels_list.append(0)
        else:
            labels_list.append(1)
        times_list.append(v[1])

    labels_list = np.array(labels_list)
    times_list = np.array(times_list)

    # 使用 numpy.vstack 将所有数组垂直堆叠起来
    # 如果数组的维度相同，也可以使用 numpy.concatenate(features_list, axis=0)
    test_features = np.concatenate(features_list, axis=0)

    test_labels = labels_list
    time_labels = times_list

    # test_labels = np.concatenate(labels_list, axis=0)
    # time_labels = np.concatenate(times_list, axis=0)

    test_features = torch.tensor(test_features, dtype=torch.float32).squeeze()
    test_labels = torch.tensor(test_labels, dtype=torch.float32)
    time_labels = torch.tensor(time_labels, dtype=torch.float32)

    dataset = TensorDataset(test_features, test_labels)

    # indices = torch.randperm(len(dataset))
    # split_idx = int(0.8 * len(dataset))
    #
    # test_dataset = Subset(dataset, indices[:split_idx])
    # test_dataset = Subset(dataset, indices[split_idx:])

    batch_size = 64
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ...测试集评估代码...
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, labels in test_loader:
            outputs = model(data)
            predicted = (outputs > 0.5).float()  # Convert logits to predictions (0 or 1)
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()

        print(f'Accuracy of the model on the test images: {100 * correct / total} %')





    # items = list(result_dict.items())
    # random.shuffle(items)
    # result_dict = dict(items)
    # for key, value in result_dict.items():
    #     list1 = solve_dict[key]
    #     file_path = key
    #
    #     with open(file_path, 'r') as file:
    #         # 读取文件所有内容到一个字符串
    #         smtlib_str = file.read()
    #     # # 解析字符串
    #     # try:
    #     #     # 将JSON字符串转换为字典
    #     #     dict_obj = json.loads(smtlib_str)
    #     #     # print("转换后的字典：", dict_obj)
    #     # except json.JSONDecodeError as e:
    #     #     print("解析错误：", e)
    #     # #
    #     # if 'smt-comp' in file_path:
    #     #     smtlib_str = dict_obj['smt_script']
    #     # else:
    #     #     smtlib_str = dict_obj['script']
    #     # variables = set()
    #     # variables = extract_variables_from_smt2_content(smtlib_str)
    #     # smtlib_str = normalize_variables(smtlib_str, variables)
    #     smtlib_str, sorted_variable_dict,_ = normalize_smt_str(smtlib_str)
    #     v = torch.tensor(json.load(value[0]), dtype=torch.float32).squeeze()
    #     # v = embedder.get_max_pooling_embedding(smtlib_str,sorted_variable_dict)
    #     output = model(v)
    #     print('output:', (output > 0.5).int().item())

def spilt_files():
    import random
    with open('/home/lz/PycharmProjects/Pearl/test_rl/predictor/embeding_QF_IDL.json', 'r') as file:
        result_dict = json.load(file)
    with open('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_smt_comp.txt', 'r') as file:
        solve_dict = json.load(file)

    for k,v in result_dict.items():
        if k in solve_dict:
            if solve_dict[k][1] > 300:
                # result_dict[k] = v
                print(k,solve_dict[k])
    # 将字典转换为列表，以便随机化
    data_list = list(result_dict.items())

    # 随机打乱列表
    random.shuffle(data_list)

    # 计算一半的数据量
    half_size = len(data_list) // 2

    # 分割数据为训练集和测试集
    train_set = dict(data_list[:half_size])
    test_set = dict(data_list[half_size:])

    # 打印结果
    # print("训练集:", train_set)
    # print("测试集:", test_set)
    print(len(train_set),len(test_set))
    with open('QF_IDF_train.json', 'w') as file:
        json.dump(train_set, file, indent=4)
    with open('QF_IDF_test.json', 'w') as file:
        json.dump(test_set, file, indent=4)
if __name__ == '__main__':
    # train()
    test()
    # spilt_files()
