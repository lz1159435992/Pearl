import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel

# 初始化分词器和模型
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model = RobertaModel.from_pretrained('microsoft/codebert-base')

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

def embed_code(code, tokenizer, model, max_length=512):
    inputs = tokenizer.encode_plus(code, add_special_tokens=True, max_length=max_length,
                                   padding='max_length', truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state

def update_embedding_with_new_code(original_embedding, new_code, tokenizer, model, attention_pooling, max_length=512):
    # 嵌入新代码段
    new_embedding = embed_code(new_code, tokenizer, model, max_length)

    # 合并原始嵌入和新嵌入
    combined_embedding = torch.cat((original_embedding, new_embedding), dim=1)

    # 使用注意力机制融合合并后的嵌入
    updated_embedding = attention_pooling(combined_embedding)
    return updated_embedding

# 示例
original_code = "(set-logic QF_AUFBV )\n(declare-fun arg00 () (Array (_ BitVec 32) (_ BitVec 8) ) )\n(assert (let ( (?B1 (select  arg00 (_ bv3 32) ) ) (?B2 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv8 32) ) ) ) ) ) (?B3 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv9 32) ) ) ) ) ) (?B4 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv7 32) ) ) ) ) ) (?B5 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv6 32) ) ) ) ) ) (?B6 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv4 32) ) ) ) ) ) (?B7 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv5 32) ) ) ) ) ) ) (let ( (?B8 ((_ sign_extend 24)  ?B1 ) ) ) (let ( (?B9 ((_ extract 7  0)  (bvadd  (_ bv4294967209 32) (bvor  ?B8 (_ bv32 32) ) ) ) ) ) (and  (and  (and  (and  (and  (and  (and  (and  (and  (and  (=  (_ bv0 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) ((_ zero_extend 56)  ?B9 ) ) ((_ zero_extend 56)  ?B6 ) ) ) ((_ zero_extend 56)  ?B7 ) ) ) ((_ zero_extend 56)  ?B5 ) ) ) ((_ zero_extend 56)  ?B4 ) ) ) ((_ zero_extend 56)  ?B2 ) ) ) ((_ zero_extend 56)  ?B3 ) ) ) (=  false (=  (_ bv48 8) ?B1 ) ) ) (=  false (bvsle  ((_ zero_extend 24)  ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ?B8 ) ) ) (_ bv9 32) ) ) ) (bvsle  (_ bv65 32) ?B8 ) ) (=  false (bvsle  (_ bv10 32) ((_ zero_extend 24)  ?B9 ) ) ) ) (bvsle  ((_ zero_extend 24)  ?B6 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B7 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B5 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B4 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B2 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B3 ) (_ bv9 32) ) ) ) ) ) )\n(check-sat)\n(get-value ( (select arg00 (_ bv0 32) ) ) )\n(get-value ( (select arg00 (_ bv1 32) ) ) )\n(get-value ( (select arg00 (_ bv2 32) ) ) )\n(get-value ( (select arg00 (_ bv3 32) ) ) )\n(get-value ( (select arg00 (_ bv4 32) ) ) )\n(get-value ( (select arg00 (_ bv5 32) ) ) )\n(get-value ( (select arg00 (_ bv6 32) ) ) )\n(get-value ( (select arg00 (_ bv7 32) ) ) )\n(get-value ( (select arg00 (_ bv8 32) ) ) )\n(get-value ( (select arg00 (_ bv9 32) ) ) )\n(get-value ( (select arg00 (_ bv10 32) ) ) )\n(exit)\n"  # 这里放置您的长代码

new_code = "(set-logic QF_AUFBV )\n(declare-fun arg00 () (Array (_ BitVec 32) (_ BitVec 8) ) )\n(assert (let ( (?B1 (select  arg00 (_ bv3 32) ) ) (?B2 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv8 32) ) ) ) ) ) (?B3 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv9 32) ) ) ) ) ) (?B4 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv7 32) ) ) ) ) ) (?B5 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv6 32) ) ) ) ) ) (?B6 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv4 32) ) ) ) ) ) (?B7 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv5 32) ) ) ) ) ) ) (let ( (?B8 ((_ sign_extend 24)  ?B1 ) ) ) (let ( (?B9 ((_ extract 7  0)  (bvadd  (_ bv4294967209 32) (bvor  ?B8 (_ bv32 32) ) ) ) ) ) (and  (and  (and  (and  (and  (and  (and  (and  (and  (and  (=  (_ bv0 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) ((_ zero_extend 56)  ?B9 ) ) ((_ zero_extend 56)  ?B6 ) ) ) ((_ zero_extend 56)  ?B7 ) ) ) ((_ zero_extend 56)  ?B5 ) ) ) ((_ zero_extend 56)  ?B4 ) ) ) ((_ zero_extend 56)  ?B2 ) ) ) ((_ zero_extend 56)  ?B3 ) ) ) (=  false (=  (_ bv48 8) ?B1 ) ) ) (=  false (bvsle  ((_ zero_extend 24)  ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ?B8 ) ) ) (_ bv9 32) ) ) ) (bvsle  (_ bv65 32) ?B8 ) ) (=  false (bvsle  (_ bv10 32) ((_ zero_extend 24)  ?B9 ) ) ) ) (bvsle  ((_ zero_extend 24)  ?B6 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B7 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B5 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B4 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B2 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B3 ) (_ bv9 32) ) ) ) ) ) )\n(check-sat)\n(get-value ( (select arg00 (_ bv0 32) ) ) )\n(get-value ( (select arg00 (_ bv1 32) ) ) )\n(get-value ( (select arg00 (_ bv2 32) ) ) )\n(get-value ( (select arg00 (_ bv3 32) ) ) )\n(get-value ( (select arg00 (_ bv4 32) ) ) )\n(get-value ( (select arg00 (_ bv5 32) ) ) )\n(get-value ( (select arg00 (_ bv6 32) ) ) )\n(get-value ( (select arg00 (_ bv7 32) ) ) )\n(get-value ( (select arg00 (_ bv8 32) ) ) )\n(get-value ( (select arg00 (_ bv9 32) ) ) )\n(get-value ( (select arg00 (_ bv10 32) ) ) )\n(exit)\n"  # 这里放置您的长代码


# 嵌入原始代码
original_embedding = embed_code(original_code, tokenizer, model)

# 初始化注意力池化层
attention_pooling = AttentionPooling(model.config.hidden_size)

# 更新嵌入
updated_embedding = update_embedding_with_new_code(original_embedding, new_code, tokenizer, model, attention_pooling)

print(updated_embedding.shape)