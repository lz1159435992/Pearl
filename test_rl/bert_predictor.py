import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Subset

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

train_features = np.load('features.npy')
train_labels = np.load('labels.npy')
time_labels = np.load('time.npy')

train_features = torch.tensor(train_features, dtype=torch.float32).squeeze()
train_labels = torch.tensor(train_labels, dtype=torch.float32)
time_labels = torch.tensor(time_labels, dtype=torch.float32)

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for data, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader)}')

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

torch.save(model.state_dict(), 'bert_predictor.pth')