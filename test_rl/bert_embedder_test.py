import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel


class CodeEmbedder_normalize:
    def __init__(self, model_name='/home/lz/baidudisk/codebert-base', max_length=512, chunk_size=128, overlap_size=32):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def get_attention_pooling(self, hidden_states, attention_mask):
        # Apply attention mechanism to get weights
        attention_scores = torch.matmul(hidden_states, hidden_states.transpose(-1, -2))
        attention_scores = attention_scores / (hidden_states.size(-1) ** 0.5)
        attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # Apply attention weights to hidden states
        attention_output = torch.matmul(attention_weights, hidden_states)
        return attention_output.mean(dim=1)

    def get_max_pooling_embedding(self, code, var_dict):
        #处理不同数据类型
        if isinstance(var_dict, dict):
            var_list = [v for k, v in var_dict.items()]
        elif isinstance(var_dict, list):
            var_list = var_dict
        tokens = self.tokenizer.encode_plus(
            code,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_tensors='pt'
        )['input_ids'].squeeze(0)

        embeddings = []
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk = tokens[start:end]

            if len(chunk) < self.chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, self.chunk_size - len(chunk)),
                                                value=self.tokenizer.pad_token_id)

            attention_mask = torch.where(chunk == self.tokenizer.pad_token_id, 0, 1)
            inputs = {'input_ids': chunk.unsqueeze(0), 'attention_mask': attention_mask.unsqueeze(0)}

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Use attention pooling instead of max pooling
            attention_pooling_embedding = self.get_attention_pooling(outputs.last_hidden_state,
                                                                     attention_mask.unsqueeze(0))
            embeddings.append(attention_pooling_embedding)

            start += self.chunk_size - self.overlap_size

        embeddings = torch.stack(embeddings).mean(dim=0)

        # embeddings = torch.stack(embeddings).mean(dim=0)
        combined_var_embed = torch.zeros_like(embeddings)
        variable_ids = [self.tokenizer.encode(variable, add_special_tokens=False) for variable in var_list]
        #获取ids的维度
        var_len = len(variable_ids[0])
        embed_len = embeddings.size(1)
        if len(variable_ids) > 0:
            variable_embeddings = []
            for n, ids in enumerate(variable_ids):
                ids = torch.tensor(ids, dtype=torch.long)
                with torch.no_grad():
                    embedded = self.model.embeddings.word_embeddings(ids)
                variable_mask = self.create_dynamic_mask(n, embed_len,len(ids), sigma=5)
                combined_var_embed = combined_var_embed + torch.mean(embedded * variable_mask.T)
                # variable_embeddings.append(embedded)
            # for n, (var_embeds, ids) in enumerate(zip(variable_embeddings, variable_ids)):
            #     if len(ids) == 0:
            #         continue
            #     variable_mask = self.create_dynamic_mask(n, embeddings.size(1), sigma=5)
            #     print(var_embeds.shape, variable_mask.shape)
            #     # print((var_embeds*variable_mask).shape)
            #     combined_var_embed = combined_var_embed + torch.mean(var_embeds*variable_mask.T)

        embeddings = embeddings + combined_var_embed

        return embeddings

    def create_dynamic_mask(self, n, embed_len, var_len, sigma=5):
        mask = torch.zeros((embed_len, var_len), dtype=torch.float32)
        for i in range(var_len):
            mask[:, i] += torch.exp(-(torch.arange(embed_len, dtype=torch.float32) - n) ** 2 / (2 * sigma ** 2))
        mask = mask / mask.max()
        return mask
