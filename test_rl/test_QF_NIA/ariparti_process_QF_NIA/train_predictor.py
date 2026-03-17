"""
QF_NIA预测器模型训练模块（AriParti 版本）
包含二分类模型(可解性预测)和八分类模型(求解时间预测)
保持与 cvc5 版本一致的模型与训练流程，输出路径在当前目录 models/ 下。
"""
import json
import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from loguru import logger

# 保证项目根路径可导入
import sys
if '/home/lz/PycharmProjects/Pearl' not in sys.path:
    sys.path.insert(0, '/home/lz/PycharmProjects/Pearl')

from test_rl.test_script.utils import setup_logger

if __name__ == "__main__":
    setup_logger()

# ==================== 模型定义 ====================

class ResBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)

    def forward(self, x):
        identity = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out) + identity
        return F.relu(out)


class EnhancedClassifier(nn.Module):
    def __init__(self, input_dim=8192):
        super(EnhancedClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.resblock1 = ResBlock(2048, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.resblock2 = ResBlock(512, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.resblock1(x)
        x = F.relu(self.fc2(x))
        x = self.resblock2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class EnhancedEightClassModelLargeInput(nn.Module):
    def __init__(self, input_dim=8192):
        super(EnhancedEightClassModelLargeInput, self).__init__()
        self.input_size = input_dim
        self.fc1 = nn.Linear(self.input_size, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(512, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(128, 8)
        self.residual_fc = nn.Linear(self.input_size, 8)

    def forward(self, x):
        residual = self.residual_fc(x)
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = F.leaky_relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        x = self.fc5(x)
        return x + residual


# ==================== 数据准备函数 ====================

def prepare_train_val_split(train_json=None, train_val_split=0.8, random_seed=42):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if train_json is None:
        train_json = os.path.join(script_dir, 'QF_NIA_train.json')

    logger.info('='*50)
    logger.info('准备数据集划分')
    logger.info('='*50)
    logger.info(f'训练数据: {train_json}')
    logger.info(f'训练/验证集划分: {train_val_split*100:.0f}% / {(1-train_val_split)*100:.0f}%')
    logger.info(f'随机种子: {random_seed}')

    with open(train_json, 'r') as f:
        train_dict = json.load(f)

    valid_keys = [k for k, v in train_dict.items() if len(v) >= 3 and os.path.exists(v[0])]
    total_samples = len(valid_keys)
    logger.info(f'总样本数: {total_samples}')

    torch.manual_seed(random_seed)
    indices = torch.randperm(total_samples)
    split_idx = int(train_val_split * total_samples)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    logger.info(f'训练集大小: {len(train_indices)} ({len(train_indices)/total_samples*100:.1f}%)')
    logger.info(f'验证集大小: {len(val_indices)} ({len(val_indices)/total_samples*100:.1f}%)')
    logger.info('='*50)

    return train_indices, val_indices, train_dict


# ==================== 训练和验证函数 ====================

def validate_binary(model, test_loader, criterion, device='cpu', log=True):
    model.eval(); model.to(device)
    total_loss = 0; correct = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device); targets = targets.to(device)
            outputs = model(data)
            loss = criterion(outputs.squeeze(), targets)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted.squeeze() == targets).sum().item()
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    if log:
        logger.info(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
        print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy


def validate_multiclass(model, test_loader, criterion, device='cpu', log=True):
    model.eval(); model.to(device)
    total_loss = 0; correct = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device); targets = targets.to(device).long()
            outputs = model(data)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader); accuracy = correct / len(test_loader.dataset)
    if log:
        logger.info(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
        print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy


def train_binary_classifier_QF_NIA(
    train_indices=None,
    val_indices=None,
    train_dict=None,
    save_path=None,
    num_epochs=200,
    batch_size=64,
    learning_rate=0.001,
    early_stopping_patience=10,
    device='cuda'
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if save_path is None:
        save_path = os.path.join(script_dir, 'models', 'QF_NIA_bert_predictor_mask_best.pth')

    logger.info('='*50)
    logger.info('开始训练二分类模型（可解性预测）')
    logger.info('='*50)
    logger.info(f'模型保存路径: {save_path}')
    logger.info(f'训练集大小: {len(train_indices)}')
    logger.info(f'验证集大小: {len(val_indices)}')

    all_features = []
    all_labels = []
    for file_path, info in train_dict.items():
        if len(info) >= 2 and os.path.exists(info[0]):
            embedding = np.load(info[0])
            all_features.append(embedding)
            all_labels.append(info[1])

    all_features = torch.tensor(np.array(all_features), dtype=torch.float32).squeeze()
    all_labels = torch.tensor(all_labels, dtype=torch.float32)

    dataset = TensorDataset(all_features, all_labels)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = EnhancedClassifier(input_dim=8192)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    device_t = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device_t)
    logger.info(f'使用设备: {device_t}')

    best_val_loss = float('inf'); best_accuracy = 0.0; no_improve_epochs = 0
    for epoch in range(num_epochs):
        model.train(); epoch_loss = 0
        for data, labels in train_loader:
            data = data.to(device_t); labels = labels.to(device_t)
            optimizer.zero_grad(); outputs = model(data)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward(); optimizer.step(); epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')
        val_loss, val_accuracy = validate_binary(model, val_loader, criterion, device_t)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss; best_accuracy = val_accuracy; no_improve_epochs = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f'模型已保存: {save_path} (Loss: {best_val_loss:.4f}, Acc: {best_accuracy:.4f})')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                logger.info(f"早停: 第 {epoch + 1} 轮"); break

    logger.info('='*50)
    logger.info(f'训练完成！最佳验证损失: {best_val_loss:.4f}, 最佳准确率: {best_accuracy:.4f}')
    logger.info('='*50)
    return model


def train_eight_class_model_QF_NIA(
    train_indices=None,
    val_indices=None,
    train_dict=None,
    save_path=None,
    num_epochs=400,
    batch_size=64,
    learning_rate=1e-3,
    early_stopping_patience=20,
    device='cuda'
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if save_path is None:
        save_path = os.path.join(script_dir, 'models', 'QF_NIA_bert_predictor_2_mask_best_model.pth')

    logger.info('='*50)
    logger.info('开始训练八分类模型（求解时间预测）')
    logger.info('='*50)
    logger.info(f'模型保存路径: {save_path}')
    logger.info(f'训练集大小: {len(train_indices)}')
    logger.info(f'验证集大小: {len(val_indices)}')

    all_features = []
    all_labels = []
    for file_path, info in train_dict.items():
        if len(info) >= 3 and os.path.exists(info[0]):
            embedding = np.load(info[0])
            all_features.append(embedding)
            all_labels.append(info[2])

    all_features = torch.tensor(np.array(all_features), dtype=torch.float32).squeeze()
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    dataset = TensorDataset(all_features, all_labels)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = EnhancedEightClassModelLargeInput(input_dim=8192)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    device_t = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device_t)
    logger.info(f'使用设备: {device_t}')

    best_val_loss = float('inf'); best_accuracy = 0.0; no_improve_epochs = 0
    for epoch in range(num_epochs):
        model.train(); epoch_loss = 0
        for data, labels in train_loader:
            data = data.to(device_t); labels = labels.to(device_t)
            optimizer.zero_grad(); outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step(); epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')
        val_loss, val_accuracy = validate_multiclass(model, val_loader, criterion, device_t)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss; best_accuracy = val_accuracy; no_improve_epochs = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f'模型已保存: {save_path} (Loss: {best_val_loss:.4f}, Acc: {best_accuracy:.4f})')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                logger.info(f"早停: 第 {epoch + 1} 轮"); break

    logger.info('='*50)
    logger.info(f'训练完成！最佳验证损失: {best_val_loss:.4f}, 最佳准确率: {best_accuracy:.4f}')
    logger.info('='*50)
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='训练QF_NIA预测器模型（AriParti 版本）')
    parser.add_argument('--mode', type=str, choices=['binary', 'multiclass', 'both'], default='both')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--train_val_split', type=float, default=0.8)
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    logger.info('='*70)
    logger.info('准备训练数据（统一划分，两个模型共享）')
    logger.info('='*70)
    train_indices, val_indices, train_dict = prepare_train_val_split(
        train_val_split=args.train_val_split,
        random_seed=args.random_seed
    )

    if args.mode in ['binary', 'both']:
        logger.info('\n训练二分类模型...')
        train_binary_classifier_QF_NIA(
            train_indices=train_indices,
            val_indices=val_indices,
            train_dict=train_dict,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device
        )

    if args.mode in ['multiclass', 'both']:
        logger.info('\n训练八分类模型...')
        train_eight_class_model_QF_NIA(
            train_indices=train_indices,
            val_indices=val_indices,
            train_dict=train_dict,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device
        )

    logger.info('='*70)
    logger.info('所有模型训练完成！')
    logger.info('='*70)
