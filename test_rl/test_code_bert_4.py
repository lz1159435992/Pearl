from transformers import RobertaTokenizer, RobertaModel
import torch
from torch.nn.functional import cosine_similarity
from test_script.utils import extract_variables_from_smt2_content

class CodeEmbedder:
    def __init__(self, model_name='/home/lz/baidudisk/codebert-base', max_length=512, chunk_size=512):
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
class CodeEmbedder_normalize:
    def __init__(self, model_name='/home/lz/baidudisk/codebert-base', max_length=512, chunk_size=512):
        # 初始化分词器和模型
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.max_length = max_length
        self.chunk_size = chunk_size

    def get_max_pooling_embedding(self, code):
        # 分块处理代码
        vars = extract_variables_from_smt2_content(code)
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

        #计算所有块的平均嵌入
        embeddings = torch.stack(embeddings).mean(dim=0)

        # #添加变量嵌入
        # # 创建变量顺序的mask，并将对应的变量嵌入添加到原始编码中
        # variable_ids = [self.tokenizer.encode(variable)[1:-1] for variable in vars]
        # variable_ids = [item for sublist in variable_ids for item in sublist] # 将多个列表展开为一个列表
        # variable_ids = torch.tensor(variable_ids, dtype=torch.long)
        # variable_mask = torch.zeros_like(embeddings)
        # variable_mask[:, :len(variable_ids)] = 1
        # variable_embedding = self.model.embeddings.word_embeddings(variable_ids.clone().detach().unsqueeze(0))
        # updated_embedding = embeddings * (1 - variable_mask) + variable_embedding * variable_mask
        # print(updated_embedding.shape)

        variable_ids = [self.tokenizer.encode(variable)[1:-1] for variable in vars]
        variable_ids = [item for sublist in variable_ids for item in sublist]
        variable_ids = torch.tensor(variable_ids, dtype=torch.long)
        variable_mask = torch.zeros_like(embeddings)
        variable_mask[:, :len(variable_ids)] = 1
        variable_embedding = self.model.embeddings.word_embeddings(variable_ids.clone().detach().unsqueeze(0))
        updated_embedding = embeddings + torch.sum(variable_embedding * variable_mask, dim=1)
        print(updated_embedding.shape)
        return updated_embedding
    #更新变量嵌入
    def update_embedding(self, existing_embedding, new_code, variable_list, weight=0.5):
        # 获取新代码的嵌入
        new_code_embedding = self.get_max_pooling_embedding(new_code)

        # 添加变量顺序信息
        variable_ids = [self.tokenizer.encode(variable)[1:-1] for variable in variable_list]
        variable_ids = [item for sublist in variable_ids for item in sublist]  # 将多个列表展开为一个列表
        variable_order_embedding = self.model.roberta.embeddings.word_embeddings(
            torch.tensor(variable_ids).unsqueeze(0))

        # 使用加权平均更新嵌入
        updated_embedding = (1 - weight) * existing_embedding + weight * new_code_embedding + variable_order_embedding
        return updated_embedding
    def update_embedding_newcode(self, existing_embedding, new_code, weight=0.5):
        # 获取新代码的嵌入
        new_code_embedding = self.get_max_pooling_embedding(new_code)

        # 使用加权平均更新嵌入
        updated_embedding = (1 - weight) * existing_embedding + weight * new_code_embedding
        return updated_embedding
# 示例使用
# code_embedder = CodeEmbedder()
#
# existing_code = 'int var_2 = 0'
# existing_embedding = code_embedder.get_max_pooling_embedding(existing_code)
# existing_code = 'var_2'
# existing_embedding_2 = code_embedder.get_max_pooling_embedding(existing_code)
# print(torch.allclose(existing_embedding, existing_embedding_2, atol=1e-03))
#
# cosine_sim = cosine_similarity(existing_embedding , existing_embedding_2)
#
# print(f"Cosine similarity: {cosine_sim.item()}")
# euclidean_distance = torch.norm(existing_embedding - existing_embedding_2)
#
# # 曼哈顿距离
# manhattan_distance = torch.abs(existing_embedding_2).sum(dim=-1)
#
# print(f"Euclidean distance: {euclidean_distance.item()}")
# print(f"Manhattan distance: {manhattan_distance.item()}")
# # 更新嵌入，可以根据需要调整weight
# updated_embedding = code_embedder.update_embedding(existing_embedding, new_code, weight=0.2)
# print(updated_embedding)
# print(updated_embedding.shape)  # 输出保持不变的固定维度
