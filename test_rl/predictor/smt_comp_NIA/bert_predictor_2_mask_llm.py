import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from transformers import BertModel, BertTokenizer

from test_rl.test_script.utils import setup_logger
from loguru import logger
setup_logger()
def to_one_hot(indices, num_classes):
    """
    Convert a numpy array of indices to a one-hot encoded numpy array.

    Parameters:
    - indices: numpy array of labels (integers representing the class index)
    - num_classes: total number of classes

    Returns:
    - one_hot_array: a one-hot encoded numpy array
    """
    one_hot_array = np.eye(num_classes)[indices]
    return one_hot_array


def train(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    model.to(device)  # 将模型移动到 GPU

    best_val_loss = float('inf')
    no_improve_epochs = 0
    early_stopping_patience = 20

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data, targets in train_loader:
            data = data.to(device)         # 将输入数据移到 GPU
            targets = targets.to(device)   # 将标签移到 GPU

            optimizer.zero_grad()
            outputs = model(data)
            targets = targets.long()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_train_loss}')
        logger.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_train_loss}')

        # 验证
        val_loss = validate(model, test_loader, criterion, log=True, device=device)

        # 学习率调度器
        scheduler.step(val_loss)

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'QF_NIA_bert_predictor_2_mask_best_model_llm.pth')
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # 保存最终模型
    torch.save(model.state_dict(), 'QF_NIA_bert_predictor_2_mask_final_model_llm.pth')


def validate(model, test_loader, criterion, log=True, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)  # 确保模型在正确的设备上
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            targets = targets.long()
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)

    if log:
        print(f'Validation Loss: {avg_loss}, Accuracy: {accuracy}')
        logger.info(f'Validation Loss: {avg_loss}, Accuracy: {accuracy}')

    return avg_loss

class EnhancedEightClassModelLargeInput(nn.Module):
    def __init__(self):
        super(EnhancedEightClassModelLargeInput, self).__init__()
        self.input_size = 8192  # 调整输入维度

        # 第一层：从 8192 降到 2048
        self.fc1 = nn.Linear(self.input_size, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(0.5)

        # 第二层：从 2048 降到 1024
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.5)

        # 第三层：从 1024 降到 512
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.5)

        # 第四层：从 512 降到 128
        self.fc4 = nn.Linear(512, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.5)

        # 第五层：从 128 降到 8 (分类任务)
        self.fc5 = nn.Linear(128, 8)

        # 添加残差连接，直接将 8192 映射到 8
        self.residual_fc = nn.Linear(self.input_size, 8)

    def forward(self, x):
        # 残差连接
        residual = self.residual_fc(x)

        # 前向传播
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = F.leaky_relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)

        x = self.fc5(x)

        # 将残差连接叠加到输出
        return x + residual

class EnhancedEightClassModel(nn.Module):
    def __init__(self):
        super(EnhancedEightClassModel, self).__init__()
        self.input_size = 768

        self.fc1 = nn.Linear(self.input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(64, 8)

        # 添加残差连接
        self.residual_fc = nn.Linear(self.input_size, 8)

    def forward(self, x):
        residual = self.residual_fc(x)  # 残差连接

        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)

        return x + residual
class ImprovedEightClassModel(nn.Module):
    def __init__(self):
        super(ImprovedEightClassModel, self).__init__()
        self.input_features = 768  # 请根据实际情况替换这个值

        self.fc1 = nn.Linear(self.input_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(64, 8)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


class EightClassModel(nn.Module):
    def __init__(self):
        super(EightClassModel, self).__init__()
        # 假设输入特征的数量为input_features
        self.input_features = 768  # 请根据实际情况替换这个值
        # 定义第一个全连接层
        self.fc1 = nn.Linear(self.input_features, 128)
        # 定义第二个全连接层
        self.fc2 = nn.Linear(128, 64)
        # 定义输出层，输出8个神经元对应8个类别
        self.fc3 = nn.Linear(64, 8)

    def forward(self, x):
        # 通过第一个全连接层
        x = F.relu(self.fc1(x))
        # 通过第二个全连接层
        x = F.relu(self.fc2(x))
        # 通过输出层，不使用激活函数，因为分类任务的输出是概率分布
        x = self.fc3(x)
        return x

def training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open('QF_NIA_train.json', 'r') as file:
        train_dict = json.load(file)

    with open('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/NIA/NIA.json', 'r') as file:
        solve_dict = json.load(file)

    features_list = []
    labels_list = []
    times_list = []

    for k, v in train_dict.items():
        features_list.append(np.load(v[0]))
        labels_list.append(v[1])
        times_list.append(v[2])

    labels_list = np.array(labels_list)
    times_list = np.array(times_list)

    train_features = np.concatenate(features_list, axis=0)
    train_labels = labels_list
    time_labels = times_list

    train_features = torch.tensor(train_features, dtype=torch.float32).squeeze()
    train_features = train_features.view(-1, 8192)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    time_labels = torch.tensor(time_labels, dtype=torch.long)  # 注意类型

    dataset = TensorDataset(train_features, time_labels)

    indices = torch.randperm(len(dataset))
    split_idx = int(0.8 * len(dataset))

    train_dataset = Subset(dataset, indices[:split_idx])
    test_dataset = Subset(dataset, indices[split_idx:])

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = EnhancedEightClassModelLargeInput().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    num_epochs = 400
    train(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device)

    model.load_state_dict(torch.load('QF_NIA_bert_predictor_2_mask_best_model_llm.pth', map_location=device))
    validate(model, test_loader, criterion, device)
def testing():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open('QF_NIA_test.json', 'r') as file:
        test_dict = json.load(file)

    with open('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/NIA/NIA.json', 'r') as file:
        solve_dict = json.load(file)

    features_list = []
    labels_list = []
    times_list = []

    for k, v in test_dict.items():
        features_list.append(np.load(v[0]))
        labels_list.append(v[1])
        times_list.append(v[2])

    # 合并所有特征和标签
    test_features = np.concatenate(features_list, axis=0)
    test_labels = np.array(labels_list)
    time_labels = np.array(times_list)

    # 转换为 Tensor
    test_features = torch.tensor(test_features, dtype=torch.float32).squeeze()
    test_features = test_features.view(-1, 8192)  # reshape 为 (n_samples, 8192)
    time_labels = torch.tensor(time_labels, dtype=torch.long)  # CrossEntropyLoss 需要 long 类型

    # 创建数据集和 DataLoader
    dataset = TensorDataset(test_features, time_labels)

    indices = torch.randperm(len(dataset))
    split_idx = int(0.8 * len(dataset))

    train_dataset = Subset(dataset, indices[:split_idx])
    test_dataset = Subset(dataset, indices[split_idx:])

    batch_size = 64
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型并移动到 GPU
    model = EnhancedEightClassModelLargeInput().to(device)
    criterion = nn.CrossEntropyLoss()

    # 加载训练好的模型（带设备映射）
    model.load_state_dict(torch.load('QF_NIA_bert_predictor_2_mask_best_model_llm.pth', map_location=device))

    # 在测试集上评估模型
    validate(model, test_loader, criterion, device=device)
if __name__ == '__main__':
    training()
    # testing()