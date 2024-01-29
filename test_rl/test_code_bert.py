from transformers import AutoTokenizer, AutoModel, RobertaTokenizer, RobertaModel
import torch
import torch.nn as nn
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
nl_tokens = tokenizer.tokenize("return maximum value")

code_tokens = tokenizer.tokenize("(set-logic QF_AUFBV )\n(declare-fun arg00 () (Array (_ BitVec 32) (_ BitVec 8) ) )\n(assert (let ( (?B1 (select  arg00 (_ bv3 32) ) ) ")

tokens = [tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]+code_tokens+[tokenizer.eos_token]
tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
context_embeddings = model(torch.tensor(tokens_ids)[None,:])[0]
print(context_embeddings.shape)
def embed_code(code, tokenizer, model, max_length=512):
    # 将代码分割为 token
    tokens = tokenizer.tokenize(code)[:max_length-2]
    tokens = ['[CLS]'] + tokens + ['[SEP]']

    # 将 token 转换为模型的输入形式
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_tensor = torch.tensor([input_ids])

    # 获取嵌入表示
    with torch.no_grad():
        embeddings = model(input_tensor)[0]
    return embeddings
code = "(set-logic QF_AUFBV )\n(declare-fun arg00 () (Array (_ BitVec 32) (_ BitVec 8) ) )\n(assert (let ( (?B1 (select  arg00 (_ bv3 32) ) ) (?B2 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv8 32) ) ) ) ) ) (?B3 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv9 32) ) ) ) ) ) (?B4 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv7 32) ) ) ) ) ) (?B5 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv6 32) ) ) ) ) ) (?B6 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv4 32) ) ) ) ) ) (?B7 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv5 32) ) ) ) ) ) ) (let ( (?B8 ((_ sign_extend 24)  ?B1 ) ) ) (let ( (?B9 ((_ extract 7  0)  (bvadd  (_ bv4294967209 32) (bvor  ?B8 (_ bv32 32) ) ) ) ) ) (and  (and  (and  (and  (and  (and  (and  (and  (and  (and  (=  (_ bv0 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) ((_ zero_extend 56)  ?B9 ) ) ((_ zero_extend 56)  ?B6 ) ) ) ((_ zero_extend 56)  ?B7 ) ) ) ((_ zero_extend 56)  ?B5 ) ) ) ((_ zero_extend 56)  ?B4 ) ) ) ((_ zero_extend 56)  ?B2 ) ) ) ((_ zero_extend 56)  ?B3 ) ) ) (=  false (=  (_ bv48 8) ?B1 ) ) ) (=  false (bvsle  ((_ zero_extend 24)  ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ?B8 ) ) ) (_ bv9 32) ) ) ) (bvsle  (_ bv65 32) ?B8 ) ) (=  false (bvsle  (_ bv10 32) ((_ zero_extend 24)  ?B9 ) ) ) ) (bvsle  ((_ zero_extend 24)  ?B6 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B7 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B5 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B4 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B2 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B3 ) (_ bv9 32) ) ) ) ) ) )\n(check-sat)\n(get-value ( (select arg00 (_ bv0 32) ) ) )\n(get-value ( (select arg00 (_ bv1 32) ) ) )\n(get-value ( (select arg00 (_ bv2 32) ) ) )\n(get-value ( (select arg00 (_ bv3 32) ) ) )\n(get-value ( (select arg00 (_ bv4 32) ) ) )\n(get-value ( (select arg00 (_ bv5 32) ) ) )\n(get-value ( (select arg00 (_ bv6 32) ) ) )\n(get-value ( (select arg00 (_ bv7 32) ) ) )\n(get-value ( (select arg00 (_ bv8 32) ) ) )\n(get-value ( (select arg00 (_ bv9 32) ) ) )\n(get-value ( (select arg00 (_ bv10 32) ) ) )\n(exit)\n"  # 这里放置您的长代码
embeddings = embed_code(code, tokenizer, model)
print(embeddings.shape)



# 初始化分词器和模型
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model = RobertaModel.from_pretrained('microsoft/codebert-base')

def get_fixed_size_embedding(code, tokenizer, model, max_length=512):
    # 将代码编码为固定长度
    inputs = tokenizer.encode_plus(code, add_special_tokens=True, max_length=max_length,
                                   padding='max_length', truncation=True, return_tensors='pt')

    # 获取模型输出
    with torch.no_grad():
        outputs = model(**inputs)

    # 应用池化操作（这里使用平均池化）
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# 示例代码
code = "(set-logic QF_AUFBV )\n(declare-fun arg00 () (Array (_ BitVec 32) (_ BitVec 8) ) )\n(assert (let ( (?B1 (select  arg00 (_ bv3 32) ) ) (?B2 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv8 32) ) ) ) ) ) (?B3 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv9 32) ) ) ) ) ) (?B4 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv7 32) ) ) ) ) ) (?B5 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv6 32) ) ) ) ) ) (?B6 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv4 32) ) ) ) ) ) (?B7 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv5 32) ) ) ) ) ) ) (let ( (?B8 ((_ sign_extend 24)  ?B1 ) ) ) (let ( (?B9 ((_ extract 7  0)  (bvadd  (_ bv4294967209 32) (bvor  ?B8 (_ bv32 32) ) ) ) ) ) (and  (and  (and  (and  (and  (and  (and  (and  (and  (and  (=  (_ bv0 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) ((_ zero_extend 56)  ?B9 ) ) ((_ zero_extend 56)  ?B6 ) ) ) ((_ zero_extend 56)  ?B7 ) ) ) ((_ zero_extend 56)  ?B5 ) ) ) ((_ zero_extend 56)  ?B4 ) ) ) ((_ zero_extend 56)  ?B2 ) ) ) ((_ zero_extend 56)  ?B3 ) ) ) (=  false (=  (_ bv48 8) ?B1 ) ) ) (=  false (bvsle  ((_ zero_extend 24)  ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ?B8 ) ) ) (_ bv9 32) ) ) ) (bvsle  (_ bv65 32) ?B8 ) ) (=  false (bvsle  (_ bv10 32) ((_ zero_extend 24)  ?B9 ) ) ) ) (bvsle  ((_ zero_extend 24)  ?B6 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B7 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B5 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B4 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B2 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B3 ) (_ bv9 32) ) ) ) ) ) )\n(check-sat)\n(get-value ( (select arg00 (_ bv0 32) ) ) )\n(get-value ( (select arg00 (_ bv1 32) ) ) )\n(get-value ( (select arg00 (_ bv2 32) ) ) )\n(get-value ( (select arg00 (_ bv3 32) ) ) )\n(get-value ( (select arg00 (_ bv4 32) ) ) )\n(get-value ( (select arg00 (_ bv5 32) ) ) )\n(get-value ( (select arg00 (_ bv6 32) ) ) )\n(get-value ( (select arg00 (_ bv7 32) ) ) )\n(get-value ( (select arg00 (_ bv8 32) ) ) )\n(get-value ( (select arg00 (_ bv9 32) ) ) )\n(get-value ( (select arg00 (_ bv10 32) ) ) )\n(exit)\n"  # 这里放置您的长代码

embedding = get_fixed_size_embedding(code, tokenizer, model)
print(embedding)
print(embedding.shape)  # 将打印出固定维度的shap


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
        # encoder_output shape: (batch_size, seq_len, input_dim)
        attention_weights = self.attention(encoder_output)
        # attention_weights shape: (batch_size, seq_len, 1)

        pooled_output = torch.sum(encoder_output * attention_weights, dim=1)
        # pooled_output shape: (batch_size, input_dim)
        return pooled_output

def get_attention_pooling_embedding(code, tokenizer, model, max_length=512):
    # 将代码编码为固定长度
    inputs = tokenizer.encode_plus(code, add_special_tokens=True, max_length=max_length,
                                   padding='max_length', truncation=True, return_tensors='pt')

    # 获取模型输出
    with torch.no_grad():
        outputs = model(**inputs)

    # 应用注意力池化
    attention_pooling = AttentionPooling(model.config.hidden_size)
    embedding = attention_pooling(outputs.last_hidden_state)
    return embedding

# 示例代码
code = "(set-logic QF_AUFBV )\n(declare-fun arg00 () (Array (_ BitVec 32) (_ BitVec 8) ) )\n(assert (let ( (?B1 (select  arg00 (_ bv3 32) ) ) (?B2 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv8 32) ) ) ) ) ) (?B3 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv9 32) ) ) ) ) ) (?B4 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv7 32) ) ) ) ) ) (?B5 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv6 32) ) ) ) ) ) (?B6 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv4 32) ) ) ) ) ) (?B7 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv5 32) ) ) ) ) ) ) (let ( (?B8 ((_ sign_extend 24)  ?B1 ) ) ) (let ( (?B9 ((_ extract 7  0)  (bvadd  (_ bv4294967209 32) (bvor  ?B8 (_ bv32 32) ) ) ) ) ) (and  (and  (and  (and  (and  (and  (and  (and  (and  (and  (=  (_ bv0 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) ((_ zero_extend 56)  ?B9 ) ) ((_ zero_extend 56)  ?B6 ) ) ) ((_ zero_extend 56)  ?B7 ) ) ) ((_ zero_extend 56)  ?B5 ) ) ) ((_ zero_extend 56)  ?B4 ) ) ) ((_ zero_extend 56)  ?B2 ) ) ) ((_ zero_extend 56)  ?B3 ) ) ) (=  false (=  (_ bv48 8) ?B1 ) ) ) (=  false (bvsle  ((_ zero_extend 24)  ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ?B8 ) ) ) (_ bv9 32) ) ) ) (bvsle  (_ bv65 32) ?B8 ) ) (=  false (bvsle  (_ bv10 32) ((_ zero_extend 24)  ?B9 ) ) ) ) (bvsle  ((_ zero_extend 24)  ?B6 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B7 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B5 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B4 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B2 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B3 ) (_ bv9 32) ) ) ) ) ) )\n(check-sat)\n(get-value ( (select arg00 (_ bv0 32) ) ) )\n(get-value ( (select arg00 (_ bv1 32) ) ) )\n(get-value ( (select arg00 (_ bv2 32) ) ) )\n(get-value ( (select arg00 (_ bv3 32) ) ) )\n(get-value ( (select arg00 (_ bv4 32) ) ) )\n(get-value ( (select arg00 (_ bv5 32) ) ) )\n(get-value ( (select arg00 (_ bv6 32) ) ) )\n(get-value ( (select arg00 (_ bv7 32) ) ) )\n(get-value ( (select arg00 (_ bv8 32) ) ) )\n(get-value ( (select arg00 (_ bv9 32) ) ) )\n(get-value ( (select arg00 (_ bv10 32) ) ) )\n(exit)\n"  # 这里放置您的长代码


embedding = get_attention_pooling_embedding(code, tokenizer, model)
print(embedding)
print(embedding.shape)