import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, Subset
import glob
import os
from loguru import logger
from torch.cuda.amp import autocast, GradScaler

# 设置日志
logger.add("training.log", level="INFO")

# 模型定义
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ComplexClassifier(nn.Module):
    def __init__(self):
        super(ComplexClassifier, self).__init__()
        self.fc1 = nn.Linear(8192, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


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
    def __init__(self):
        super(EnhancedClassifier, self).__init__()
        self.fc1 = nn.Linear(8192, 2048)
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


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    with open('QF_NIA_train.json', 'r') as file:
        train_dict = json.load(file)
    with open('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/NIA/NIA.json', 'r') as file:
        solve_dict = json.load(file)

    features_list = []
    labels_list = []

    for k, v in train_dict.items():
        features_list.append(np.load(v[0]))
        if solve_dict[k][0] == 'sat':
            labels_list.append(0)
        else:
            labels_list.append(1)

    train_features = torch.tensor(np.concatenate(features_list, axis=0), dtype=torch.float32).view(-1, 8192)
    train_labels = torch.tensor(labels_list, dtype=torch.float32)

    dataset = TensorDataset(train_features, train_labels)
    indices = torch.randperm(len(dataset))
    split_idx = int(0.8 * len(dataset))
    train_dataset = Subset(dataset, indices[:split_idx])
    test_dataset = Subset(dataset, indices[split_idx:])
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = EnhancedClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    scaler = GradScaler()

    best_val_loss = float('inf')
    no_improve_epochs = 0
    early_stopping_patience = 10
    num_epochs = 200

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(data).squeeze()
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss}')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss}')

        # 验证
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                labels = labels.to(device)
                outputs = model(data).squeeze()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        logger.info(f'Test Accuracy: {accuracy:.2f}%')
        print(f'Test Accuracy: {accuracy:.2f}%')

        scheduler.step(avg_epoch_loss)

        if avg_epoch_loss < best_val_loss:
            best_val_loss = avg_epoch_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'QF_NIA_bert_predictor_mask_best_llm.pth')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print(f'Early stopping after {no_improve_epochs} epochs without improvement.')
                break

    torch.save(model.state_dict(), 'QF_NIA_bert_predictor_mask_final_llm.pth')


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = EnhancedClassifier().to(device)
    model_path = 'QF_NIA_bert_predictor_mask_best_llm.pth' if os.path.exists(
        'QF_NIA_bert_predictor_mask_best_llm.pth') else 'QF_NIA_bert_predictor_mask_final_llm.pth'

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    with open('QF_NIA_test.json', 'r') as file:
        test_dict = json.load(file)
    with open('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/NIA/NIA.json', 'r') as file:
        solve_dict = json.load(file)

    features_list = []
    labels_list = []

    for k, v in test_dict.items():
        features_list.append(np.load(v[0]))
        if solve_dict[k][0] == 'sat':
            labels_list.append(0)
        else:
            labels_list.append(1)

    test_features = torch.tensor(np.concatenate(features_list, axis=0), dtype=torch.float32).view(-1, 8192)
    test_labels = torch.tensor(labels_list, dtype=torch.float32)

    dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data).squeeze()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.info(f'最终结果: Accuracy on test set: {accuracy:.2f}%')
    print(f'Accuracy on test set: {accuracy:.2f}%')


if __name__ == '__main__':
    # spilt_files()
    train()
    # test()