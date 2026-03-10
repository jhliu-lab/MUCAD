import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertModel


# 定义可学习的提示
class LearnablePrompt(nn.Module):
    def __init__(self, device, num_prompt, embedding_dim):
        super(LearnablePrompt, self).__init__()
        self.device = device
        self.prompts = nn.Parameter(torch.randn(num_prompt, embedding_dim))
    def get_prompts(self):
        return self.prompts
    def get_cur_prompts(self):
        return self.get_prompts().clone().detach()
    def set_prompts(self, prompts, save_status=False):
        if save_status:
            self.prompts = prompts
        else :
            self.prompts = nn.Parameter(prompts)
    def forward(self):
        return self.prompts.to(self.device)


class TextEncoder(nn.Module):
    def __init__(self, device, model_name='bert-base-uncased', model_path='bert-base-uncased', max_length=197):
        super(TextEncoder, self).__init__()
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.backbone = BertModel.from_pretrained(model_path)
        self.max_length = max_length
        self.device = device
        self.backbone = self.backbone.to(device)


    
    def forward(self, texts, prompts=None):
        """ 
        texts: list,
        prompts: [B, N, D]
        """
        if not texts:
            raise ValueError("texts cannot be empty")
        with torch.no_grad():
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        if prompts is None:
            with torch.no_grad():
                outputs = self.backbone(**inputs)
            cls_feat = outputs.last_hidden_state[:, 0, :]
            return cls_feat
        
        # 使用分词器将句子转换为 token IDs
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        max_length = self.max_length
        combined_input_ids, combined_attention_mask = self._combine_inputs(input_ids, attention_mask, prompts, prompts.shape[0], max_length)
        combined_attention_mask = combined_attention_mask.to(self.device)
        combined_input_ids = combined_input_ids.to(self.device)
        # 获取词嵌入向量
        with torch.no_grad():
            word_embeddings = self.backbone.embeddings(combined_input_ids)
        word_embeddings = word_embeddings.to(self.device)
        # 在词嵌入向量中插入提示向量
        word_embeddings[:, input_ids.shape[1]:input_ids.shape[1] + prompts.shape[0], :] = prompts

        # 输入到 BERT 模型
        with torch.no_grad():
            # outputs = self.backbone(inputs_embeds=word_embeddings, attention_mask=combined_attention_mask)
            outputs = self.backbone(inputs_embeds = word_embeddings, attention_mask=combined_attention_mask)

        # 获取最后一层的隐藏状态
        last_hidden_state = outputs.last_hidden_state # 1*197*768
        # 返回cls token的值
        return last_hidden_state[:,0,:]

    def _combine_inputs(self, input_ids, attention_mask, prompt_embeddings, num_prompt_tokens, max_length):
        batch_size = input_ids.size(0)
        combined_input_ids = torch.full((batch_size, max_length), self.tokenizer.pad_token_id, dtype=torch.long)
        combined_attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        
        for i in range(batch_size):
            template_input_ids = input_ids[i]
            template_attention_mask = attention_mask[i]
            
            combined_input_ids[i, :len(template_input_ids)] = template_input_ids
            combined_attention_mask[i, :len(template_attention_mask)] = template_attention_mask
            
            combined_attention_mask[i, len(template_attention_mask):len(template_attention_mask) + num_prompt_tokens] = 1
        
        return combined_input_ids, combined_attention_mask


class Adapter(nn.Module):
    def __init__(self, device, input_dim, output_dim):
        super(Adapter, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                # nn.Linear(input_dim, input_dim),
                # nn.ReLU(),
                # nn.Linear(input_dim, output_dim),
                # nn.ReLU()
                ).to(device)
        self.device = device
    
    def forward(self, input):
        input = input.to(self.device, dtype=torch.float32)
        output = self.block(input)
        return output

def contrastive_sem(image_feat, text_feat):
    """
    text_feat: [1, 768]
    image_feat: [bs, 768]
    """
    norm_image_feat = F.normalize(image_feat, dim=-1)
    norm_text_feat = F.normalize(text_feat, dim=-1)
    sem_loss = - torch.mean(F.cosine_similarity(norm_image_feat, norm_text_feat))
    return sem_loss


def get_similarity_map(sm, shape):
    side = int(sm.shape[1] ** 0.5)
    sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)
    sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear')
    sm = sm.permute(0, 2, 3, 1)
    return sm


def compute_similarity(image_features, text_features, t=2):
    image_features = image_features.to(dtype = text_features.dtype) # 8*197*168
    # L2范式归一化 
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    # 余弦相似度
    similarity = image_features[:, 1:, :] @ text_features.t()
    # b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], image_features.shape[1], image_features.shape[2]
    # feats = image_features.reshape(b, n_i, 1, c) @ text_features.reshape(1, 1, n_t, c)
    # similarity = feats.sum(-1)
    # porb_mask = torch.sigmoid(similarity) # 可以将-1到1的相似度转换为0-1之间
    
    return similarity, None

def get_texts(prompt_len=1, dataset_count=0, only_text = False):
    ####################################
    dataset_name_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut','leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    # dataset_name_list = ['object', 'object', 'object', 'object', 'object', 'object','object', 'object', 'object', 'object', 'object', 'object', 'object', 'object', 'object']
    #dataset_name_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3' ,'pcb4', 'pipe_fryum']
    #dataset_name_list = ['object', 'object', 'object', 'object', 'object', 'object','object', 'object', 'object', 'object', 'object', 'object']
    ########################
    dataset_name = dataset_name_list[dataset_count]
    # if dataset_name == 'macaroni1' or dataset_name == 'macaroni2':
    #     dataset_name = 'macaroni'
    # if dataset_name in ['pcb1', 'pcb2', 'pcb3', 'pcb4']:
    #     dataset_name = 'pcb' 
    
    texts = []
    prompt_ids = []
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'normal {}']
    prompt_state = prompt_normal
    prompt_templates = [ 'a cropped photo of the {}.', 'a bright photo of a {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a good photo of the {}.', 'a photo of one {}.', 'a low resolution photo of a {}.', 'a blurry photo of a {}.', 'a photo of the small {}.', 'a photo of the large {}.', 'there is a {} in the scene.']
    prompt_sentences = []
    for i in range(len(prompt_templates)):
        for j in range(len(prompt_normal)):
            sentence = prompt_templates[i].format(prompt_state[j].format(dataset_name))
            prompt_sentences.append(sentence)
    if only_text:
        return prompt_sentences, None
            
    # normal_states = ['normal']
    # anomaly_states = ['damaged', 'broken', 'flawed']
    # src_text = "normal"
    texts.append("a photo of a " + dataset_name + " with " + "normal " * prompt_len)
    # for state in normal_states:
    #     text = src_text + f"{ state} " + dataset_name
    #     texts.append(text + " normal" * prompt_len)
    prompt_ids = list(range(7, 7 + prompt_len))
    return texts, prompt_ids 