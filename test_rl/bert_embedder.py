import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel

class CodeEmbedder_normalize:
    def __init__(self, model_name='/home/lz/baidudisk/codebert-base', max_length=512, chunk_size=512):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.max_length = max_length
        self.chunk_size = chunk_size

    def get_max_pooling_embedding(self, code, var_dict):
        vars = [v for k, v in var_dict.items()]

        tokens = self.tokenizer.encode_plus(
            code,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_tensors='pt'
        )['input_ids'].squeeze(0)

        chunks = tokens.split(self.chunk_size)

        embeddings = []

        for chunk in chunks:
            attention_mask = torch.where(chunk == self.tokenizer.pad_token_id, 0, 1)
            padded_chunk = torch.nn.functional.pad(chunk, (0, self.max_length - len(chunk)),
                                                   value=self.tokenizer.pad_token_id)
            padded_attention_mask = torch.nn.functional.pad(attention_mask, (0, self.max_length - len(chunk)), value=0)
            inputs = {'input_ids': padded_chunk.unsqueeze(0), 'attention_mask': padded_attention_mask.unsqueeze(0)}

            with torch.no_grad():
                outputs = self.model(**inputs)

            max_pooling_embedding = torch.max(outputs.last_hidden_state, dim=1)[0]
            embeddings.append(max_pooling_embedding)

        embeddings = torch.stack(embeddings).mean(dim=0)
        combined_var_embed = torch.zeros_like(embeddings)
        variable_ids = [self.tokenizer.encode(variable, add_special_tokens=False) for variable in vars]

        if len(variable_ids) > 0:
            variable_embeddings = []
            for n, ids in enumerate(variable_ids):
                ids = torch.tensor(ids, dtype=torch.long)
                with torch.no_grad():
                    embedded = self.model.embeddings.word_embeddings(ids)
                variable_mask = self.create_dynamic_mask(n, embeddings.size(1), sigma=5)
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

    def create_dynamic_mask(self, n, embed_len, sigma=5):
        mask = torch.zeros((embed_len, 3), dtype=torch.float32)
        for i in range(3):
            mask[:, i] += torch.exp(-(torch.arange(embed_len, dtype=torch.float32) - n) ** 2 / (2 * sigma ** 2))
        mask = mask / mask.max()
        return mask
