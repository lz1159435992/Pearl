import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Subset


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


train_features = np.load('features.npy')
time_labels = np.load('time.npy')
# # 将时间标签转换为one-hot编码
# time_labels = np.eye(8)[time_labels]
# print(time_labels)

train_features = torch.tensor(train_features, dtype=torch.float32).squeeze()
time_labels = torch.tensor(time_labels, dtype=torch.long)

dataset = TensorDataset(train_features, time_labels)

num_classes = 8  # Number of classes
train_labels_onehot = F.one_hot(time_labels, num_classes)

indices = torch.randperm(len(dataset))
split_idx = int(0.8 * len(dataset))

train_dataset = Subset(dataset, indices[:split_idx])
test_dataset = Subset(dataset, indices[split_idx:])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = EightClassModel()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 你可以根据需要调整学习率
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()  # 设置模型为训练模式
    for epoch in range(num_epochs):
        total_loss = 0
        for data, targets in train_loader:
            # 清除之前的梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(data)

            # 计算损失，直接使用targets，不需要one-hot编码
            loss = criterion(outputs, targets)
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            # 累加损失
            total_loss += loss.item()
        # 打印每个epoch的平均损失
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}')

# 定义验证函数
def validate(model, test_loader, criterion):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    correct = 0
    with torch.no_grad():  # 在验证过程中不计算梯度
        for data, targets in test_loader:
            outputs = model(data)
            loss = criterion(outputs, targets)  # 同样，不需要one-hot编码
            total_loss += loss.item()
            # 计算预测正确的数量
            _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
            correct += (predicted == targets).sum().item()
    # 计算并打印验证集的平均损失和准确率
    average_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / len(test_loader.dataset)
    print(f'Validation Loss: {average_loss}, Accuracy: {accuracy}%')

# 设置训练的轮数
num_epochs = 200

# 训练模型
train(model, train_loader, criterion, optimizer, num_epochs)

# 验证模型
validate(model, test_loader, criterion)

torch.save(model.state_dict(), 'bert_predictor_2.pth')