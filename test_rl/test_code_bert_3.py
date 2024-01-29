import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel

# 定义自注意力的 AttentionPooling 层
class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, encoder_output):
        attention_weights = self.attention(encoder_output)
        pooled_output = torch.sum(encoder_output * attention_weights, dim=1)
        return pooled_output

# 定义自编码器模型
class CodeAutoEncoder(nn.Module):
    def __init__(self, bert_model, attention_pooling):
        super(CodeAutoEncoder, self).__init__()
        self.bert = bert_model
        self.attention_pooling = attention_pooling
        self.decoder = nn.Linear(bert_model.config.hidden_size, bert_model.config.vocab_size)

    def forward(self, input_ids, attention_mask):
        encoder_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        pooled_output = self.attention_pooling(encoder_output)
        reconstructed_output = self.decoder(pooled_output)
        return reconstructed_output

# 初始化模型组件
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
bert_model = RobertaModel.from_pretrained('microsoft/codebert-base')
attention_pooling = AttentionPooling(bert_model.config.hidden_size)
autoencoder = CodeAutoEncoder(bert_model, attention_pooling)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

# 示例代码段
code = "(set-logic QF_AUFBV )\n(declare-fun arg00 () (Array (_ BitVec 32) (_ BitVec 8) ) )\n..."

# Tokenize and encode the code
input_ids = tokenizer.encode(code, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
attention_mask = torch.ones(input_ids.shape)

num_epochs = 3  # 训练的轮数

# 训练循环
for epoch in range(num_epochs):
    autoencoder.train()  # 将模型设置为训练模式

    optimizer.zero_grad()  # 清除之前的梯度

    # 前向传播
    output = autoencoder(input_ids=input_ids, attention_mask=attention_mask)

    # 将输出转换为与input_ids相同的形状
    output = output.view(input_ids.size())

    # 计算损失
    loss = criterion(output, input_ids.float())  # 将input_ids转换为浮点数

    # 反向传播
    loss.backward()

    # 优化器更新
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

print("训练完成")
