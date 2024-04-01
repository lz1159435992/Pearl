from transformers import RobertaTokenizer, RobertaModel
import torch

class CodeEmbedder:
    def __init__(self, model_name='/home/user/codebert-base', max_length=512, chunk_size=512):
        # 初始化分词器和模型
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.max_length = max_length
        self.chunk_size = chunk_size

    def get_max_pooling_embedding(self, code):
        # 分块处理代码
        tokens = self.tokenizer.encode_plus(code, padding=True, truncation=True,add_special_tokens=True, return_tensors='pt')['input_ids'].squeeze(0)
        chunks = tokens.split(self.chunk_size)

        # 存储每个块的最大池化嵌入
        embeddings = []

        for chunk in chunks:
            # 调整块大小并创建对应的attention_mask
            attention_mask = torch.where(chunk == self.tokenizer.pad_token_id, 0, 1)
            padded_chunk = torch.nn.functional.pad(chunk, (0, self.max_length - len(chunk)), value=self.tokenizer.pad_token_id)
            padded_attention_mask = torch.nn.functional.pad(attention_mask, (0, self.max_length - len(chunk)), value=0)
            inputs = {'input_ids': padded_chunk.unsqueeze(0), 'attention_mask': padded_attention_mask.unsqueeze(0)}

            # 获取模型输出
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 应用最大池化
            max_pooling_embedding = torch.max(outputs.last_hidden_state, dim=1)[0]
            embeddings.append(max_pooling_embedding)

        # 计算所有块的平均嵌入
        embeddings = torch.stack(embeddings).mean(dim=0)

        return embeddings

    def update_embedding(self, existing_embedding, new_code, weight=0.5):
        # 获取新代码的嵌入
        new_code_embedding = self.get_max_pooling_embedding(new_code)

        # 使用加权平均更新嵌入
        updated_embedding = (1 - weight) * existing_embedding + weight * new_code_embedding
        return updated_embedding

# 示例使用
code_embedder = CodeEmbedder()

existing_code = "(set-logic QF_AUFBV )\n(declare-fun arg00 () (Array (_ BitVec 32) (_ BitVec 8) ) )\n(assert (let ( (?B1 (select  arg00 (_ bv3 32) ) ) (?B2 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv8 32) ) ) ) ) ) (?B3 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv9 32) ) ) ) ) ) (?B4 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv7 32) ) ) ) ) ) (?B5 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv6 32) ) ) ) ) ) (?B6 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv4 32) ) ) ) ) ) (?B7 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv5 32) ) ) ) ) ) ) (let ( (?B8 ((_ sign_extend 24)  ?B1 ) ) ) (let ( (?B9 ((_ extract 7  0)  (bvadd  (_ bv4294967209 32) (bvor  ?B8 (_ bv32 32) ) ) ) ) ) (and  (and  (and  (and  (and  (and  (and  (and  (and  (and  (=  (_ bv0 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) ((_ zero_extend 56)  ?B9 ) ) ((_ zero_extend 56)  ?B6 ) ) ) ((_ zero_extend 56)  ?B7 ) ) ) ((_ zero_extend 56)  ?B5 ) ) ) ((_ zero_extend 56)  ?B4 ) ) ) ((_ zero_extend 56)  ?B2 ) ) ) ((_ zero_extend 56)  ?B3 ) ) ) (=  false (=  (_ bv48 8) ?B1 ) ) ) (=  false (bvsle  ((_ zero_extend 24)  ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ?B8 ) ) ) (_ bv9 32) ) ) ) (bvsle  (_ bv65 32) ?B8 ) ) (=  false (bvsle  (_ bv10 32) ((_ zero_extend 24)  ?B9 ) ) ) ) (bvsle  ((_ zero_extend 24)  ?B6 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B7 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B5 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B4 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B2 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B3 ) (_ bv9 32) ) ) ) ) ) )\n(check-sat)\n(get-value ( (select arg00 (_ bv0 32) ) ) )\n(get-value ( (select arg00 (_ bv1 32) ) ) )\n(get-value ( (select arg00 (_ bv2 32) ) ) )\n(get-value ( (select arg00 (_ bv3 32) ) ) )\n(get-value ( (select arg00 (_ bv4 32) ) ) )\n(get-value ( (select arg00 (_ bv5 32) ) ) )\n(get-value ( (select arg00 (_ bv6 32) ) ) )\n(get-value ( (select arg00 (_ bv7 32) ) ) )\n(get-value ( (select arg00 (_ bv8 32) ) ) )\n(get-value ( (select arg00 (_ bv9 32) ) ) )\n(get-value ( (select arg00 (_ bv10 32) ) ) )\n(exit)\n"
new_code = "(set-logic QF_AUFBV )\n(declare-fun arg00 () (Array (_ BitVec 32) (_ BitVec 8) ) )\n(assert (let ( (?B1 (select  arg00 (_ bv3 32) ) ) (?B2 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv8 32) ) ) ) ) ) (?B3 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv9 32) ) ) ) ) ) (?B4 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv7 32) ) ) ) ) ) (?B5 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv6 32) ) ) ) ) ) (?B6 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv4 32) ) ) ) ) ) (?B7 ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ((_ sign_extend 24)  (select  arg00 (_ bv5 32) ) ) ) ) ) ) (let ( (?B8 ((_ sign_extend 24)  ?B1 ) ) ) (let ( (?B9 ((_ extract 7  0)  (bvadd  (_ bv4294967209 32) (bvor  ?B8 (_ bv32 32) ) ) ) ) ) (and  (and  (and  (and  (and  (and  (and  (and  (and  (and  (=  (_ bv0 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) (bvadd  (bvmul  (_ bv10 64) ((_ zero_extend 56)  ?B9 ) ) ((_ zero_extend 56)  ?B6 ) ) ) ((_ zero_extend 56)  ?B7 ) ) ) ((_ zero_extend 56)  ?B5 ) ) ) ((_ zero_extend 56)  ?B4 ) ) ) ((_ zero_extend 56)  ?B2 ) ) ) ((_ zero_extend 56)  ?B3 ) ) ) (=  false (=  (_ bv48 8) ?B1 ) ) ) (=  false (bvsle  ((_ zero_extend 24)  ((_ extract 7  0)  (bvadd  (_ bv4294967248 32) ?B8 ) ) ) (_ bv9 32) ) ) ) (bvsle  (_ bv65 32) ?B8 ) ) (=  false (bvsle  (_ bv10 32) ((_ zero_extend 24)  ?B9 ) ) ) ) (bvsle  ((_ zero_extend 24)  ?B6 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B7 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B5 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B4 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B2 ) (_ bv9 32) ) ) (bvsle  ((_ zero_extend 24)  ?B3 ) (_ bv9 32) ) ) ) ) ) )\n(check-sat)\n(get-value ( (select arg00 (_ bv0 32) ) ) )\n(get-value ( (select arg00 (_ bv1 32) ) ) )\n(get-value ( (select arg00 (_ bv2 32) ) ) )\n(get-value ( (select arg00 (_ bv3 32) ) ) )\n(get-value ( (select arg00 (_ bv4 32) ) ) )\n(get-value ( (select arg00 (_ bv5 32) ) ) )\n(get-value ( (select arg00 (_ bv6 32) ) ) )\n(get-value ( (select arg00 (_ bv7 32) ) ) )\n(get-value ( (select arg00 (_ bv8 32) ) ) )\n(get-value ( (select arg00 (_ bv9 32) ) ) )\n(get-value ( (select arg00 (_ bv10 32) ) ) )\n(exit)\n"

existing_embedding = code_embedder.get_max_pooling_embedding(existing_code)

# 更新嵌入，可以根据需要调整weight
updated_embedding = code_embedder.update_embedding(existing_embedding, new_code, weight=0.2)
print(updated_embedding)
print(updated_embedding.shape)  # 输出保持不变的固定维度
