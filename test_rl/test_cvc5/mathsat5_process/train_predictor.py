import argparse
import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, Subset

class SimpleClassifier(nn.Module):
    """原始的二分类模型结构"""
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EnhancedEightClassModel(nn.Module):
    """原始的八分类模型结构"""
    def __init__(self):
        super(EnhancedEightClassModel, self).__init__()
        self.input_size = 768

        # 保持原有的网络结构
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

        self.residual_fc = nn.Linear(self.input_size, 8)

    def forward(self, x):
        residual = self.residual_fc(x)
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x + residual

def validate(model, test_loader, criterion, log=True):
    """验证函数"""
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                loss = criterion(outputs.squeeze(), targets)
                predicted = (outputs > 0.5).float()
                correct += (predicted.squeeze() == targets).sum().item()
            else:
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    if log:
        print(f'Validation Loss: {avg_loss}, Accuracy: {accuracy}')
    return avg_loss

def train_binary_classifier(features_dir='features/', labels_path='labels.npy', save_path=None):
    """训练二分类模型，保持原有的默认参数"""
    # 设置默认保存路径
    if save_path is None:
        save_path = 'bert_predictor_mask_best.pth'  # 原始保存路径

    # 加载特征
    file_pattern = os.path.join(features_dir, 'features_normal_*.npy')
    features_list = [np.load(f) for f in sorted(glob.glob(file_pattern),
                    key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))]
    train_features = np.concatenate(features_list, axis=0)
    train_labels = np.load(labels_path)

    # 数据准备
    train_features = torch.tensor(train_features, dtype=torch.float32).squeeze()
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    dataset = TensorDataset(train_features, train_labels)

    # 保持原有的数据集分割比例和批次大小
    indices = torch.randperm(len(dataset))
    split_idx = int(0.8 * len(dataset))
    train_dataset = Subset(dataset, indices[:split_idx])
    test_dataset = Subset(dataset, indices[split_idx:])

    batch_size = 64  # 原始批次大小
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 保持原有的模型配置
    model = SimpleClassifier()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # 训练参数
    num_epochs = 100  # 原始轮数
    best_val_loss = float('inf')
    no_improve_epochs = 0
    early_stopping_patience = 10

    # 训练循环
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader)}')

        # 验证和早停
        val_loss = validate(model, test_loader, criterion)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

def train_eight_class_model(features_dir='features/', time_labels_path='time.npy', save_path=None):
    """训练八分类模型，保持原有的默认参数"""
    # 设置默认保存路径
    if save_path is None:
        save_path = 'bert_predictor_2_mask_best_model.pth'  # 原始保存路径

    # 加载特征
    file_pattern = os.path.join(features_dir, 'features_normal_*.npy')
    features_list = [np.load(f) for f in sorted(glob.glob(file_pattern),
                    key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))]
    train_features = np.concatenate(features_list, axis=0)
    time_labels = np.load(time_labels_path)

    # 数据准备
    train_features = torch.tensor(train_features, dtype=torch.float32).squeeze()
    time_labels = torch.tensor(time_labels, dtype=torch.long)
    dataset = TensorDataset(train_features, time_labels)

    # 保持原有的数据集分割比例和批次大小
    indices = torch.randperm(len(dataset))
    split_idx = int(0.8 * len(dataset))
    train_dataset = Subset(dataset, indices[:split_idx])
    test_dataset = Subset(dataset, indices[split_idx:])

    batch_size = 64  # 原始批次大小
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 保持原有的模型配置
    model = EnhancedEightClassModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # 训练参数
    num_epochs = 400  # 原始轮数
    best_val_loss = float('inf')
    no_improve_epochs = 0
    early_stopping_patience = 10

    # 训练循环
    model.train()
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

        # 验证和早停
        val_loss = validate(model, test_loader, criterion)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

def main():
    parser = argparse.ArgumentParser(description='训练预测器模型')
    parser.add_argument('--model_type', type=str, required=True, choices=['binary', 'eight_class'],
                        help='选择训练模型类型: binary 或 eight_class')
    parser.add_argument('--features_dir', type=str, default='features/',
                        help='特征文件目录路径')
    parser.add_argument('--labels_path', type=str,
                        help='标签文件路径（binary模型用labels.npy，eight_class模型用time.npy）')
    parser.add_argument('--save_path', type=str,
                        help='模型保存路径（binary模型默认为bert_predictor_mask_best.pth，eight_class模型默认为bert_predictor_2_mask_best_model.pth）')

    args = parser.parse_args()

    # 根据模型类型选择默认的标签文件路径
    if args.labels_path is None:
        args.labels_path = 'labels.npy' if args.model_type == 'binary' else 'time.npy'

    # 根据选择调用相应的训练函数
    if args.model_type == 'binary':
        train_binary_classifier(args.features_dir, args.labels_path, args.save_path)
    else:
        train_eight_class_model(args.features_dir, args.labels_path, args.save_path)

if __name__ == '__main__':
    main() 