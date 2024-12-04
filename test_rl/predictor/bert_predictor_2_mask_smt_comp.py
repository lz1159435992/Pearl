import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from transformers import BertModel, BertTokenizer

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


def train(model, train_loader, test_loader,criterion, optimizer, scheduler, num_epochs):
    model.train()  # 设置模型为训练模式
    best_val_loss = float('inf')
    no_improve_epochs = 0
    early_stopping_patience = 10

    for epoch in range(num_epochs):
        total_loss = 0
        for data, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_train_loss}')

        # 在每个epoch结束后进行验证
        val_loss = validate(model, test_loader, criterion, log=False)

        # 调整学习率
        scheduler.step(val_loss)

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'bert_predictor_2_mask_best_model.pth')
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break


def validate(model, test_loader, criterion, log=True):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            # print('outputs:',type(outputs),outputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            # print('predicted:',type(predicted))
            # print(predicted, targets)
            correct += (predicted == targets).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    if log:
        print(f'Validation Loss: {avg_loss}, Accuracy: {accuracy}')
    return avg_loss


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
    # 定义文件名模式，这里假设文件名以 'feature_normal_' 开头并以 '.npy' 结尾
    file_pattern = 'features_normal_*.npy'

    # 使用 glob.glob 找到所有匹配的文件，并根据文件名中的数字进行排序
    file_paths = sorted(glob.glob(os.path.join('features/', file_pattern)),
                        key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))

    # 初始化一个空列表来收集所有的数组
    features_list = []

    # 逐个加载每个文件并添加到列表中
    for file_path in file_paths:
        features_list.append(np.load(file_path))

    # 使用 numpy.concatenate 将所有数组垂直堆叠起来
    train_features = np.concatenate(features_list, axis=0)

    # 加载标签
    time_labels = np.load('time.npy')

    # 将数据转换为Tensor
    train_features = torch.tensor(train_features, dtype=torch.float32).squeeze()
    time_labels = torch.tensor(time_labels, dtype=torch.long)

    dataset = TensorDataset(train_features, time_labels)

    num_classes = 8  # 类别数量
    train_labels_onehot = F.one_hot(time_labels, num_classes)

    indices = torch.randperm(len(dataset))
    split_idx = int(0.8 * len(dataset))

    train_dataset = Subset(dataset, indices[:split_idx])
    test_dataset = Subset(dataset, indices[split_idx:])

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model = EightClassModel()
    # model = ImprovedEightClassModel()
    model = EnhancedEightClassModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # 动态学习率调整
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # 早停设置
    early_stopping_patience = 10

    # 训练模型
    num_epochs = 400
    train(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs)

    # 加载最佳模型
    model.load_state_dict(torch.load('bert_predictor_2_mask_best_model.pth'))

    # 在测试集上进行最终评估
    validate(model, test_loader, criterion)
def testing():
    # 定义文件名模式，这里假设文件名以 'feature_normal_' 开头并以 '.npy' 结尾
    file_pattern = 'features_normal_*.npy'

    # 使用 glob.glob 找到所有匹配的文件，并根据文件名中的数字进行排序
    file_paths = sorted(glob.glob(os.path.join('features/', file_pattern)),
                        key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))

    # 初始化一个空列表来收集所有的数组
    features_list = []

    # 逐个加载每个文件并添加到列表中
    for file_path in file_paths:
        features_list.append(np.load(file_path))

    # 使用 numpy.concatenate 将所有数组垂直堆叠起来
    train_features = np.concatenate(features_list, axis=0)

    # 加载标签
    time_labels = np.load('time.npy')

    # 将数据转换为Tensor
    train_features = torch.tensor(train_features, dtype=torch.float32).squeeze()
    time_labels = torch.tensor(time_labels, dtype=torch.long)

    dataset = TensorDataset(train_features, time_labels)

    num_classes = 8  # 类别数量
    train_labels_onehot = F.one_hot(time_labels, num_classes)

    indices = torch.randperm(len(dataset))
    split_idx = int(0.8 * len(dataset))

    train_dataset = Subset(dataset, indices[:split_idx])
    test_dataset = Subset(dataset, indices[split_idx:])

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model = EightClassModel()
    # model = ImprovedEightClassModel()
    model = EnhancedEightClassModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # 动态学习率调整
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # 早停设置
    early_stopping_patience = 10

    # 训练模型


    # 加载最佳模型
    model.load_state_dict(torch.load('bert_predictor_2_mask_best_model.pth'))

    # 在测试集上进行最终评估
    validate(model, test_loader, criterion)
if __name__ == '__main__':
    # training()
    testing()