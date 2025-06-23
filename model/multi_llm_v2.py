import torch
import torch.nn as nn
import contextlib
import os
import torch.nn.functional as F

from opts import parse_opts
args = parse_opts()

LQ_former = args.LQ_former
use_class = args.use_class


# 定义MSEloss
criterion = nn.MSELoss()

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, num_heads=8):
        super(AttentionPooling, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads  # 每个头的维度

        # 为每个头定义一个线性层来计算注意力权重
        self.att_weights = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_dim, 1),  # 第一层线性层
                nn.ReLU(),                      # ReLU 激活层
            ) for _ in range(num_heads)
        ])

        self.linear = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        
        # 确保输入的维度正确
        assert input_dim == self.num_heads * self.head_dim, "Input dimension must be divisible by the number of heads"

        # 将输入特征划分为 num_heads 个部分，每个部分的维度为 head_dim
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        head_outputs = []
        for i in range(self.num_heads):
            # 对每个部分计算注意力权重
            attention_scores = self.att_weights[i](x[:, :, i, :])  # (batch_size, seq_len, num_heads, head_dim)
            # print('attention_scores',attention_scores.shape) #(batch_size, seq_len, 1) 
            weights = F.softmax(attention_scores, dim=1)  # softmax 在序列维度上，得到每个序列的权重

            # 使用注意力权重对每个部分进行加权求和
            head_output = (weights * x[:, :, i, :]).sum(dim=1)  # 按seq_len维度进行加权求和
            head_outputs.append(head_output)
        
        # 拼接所有头的输出，得到最终的表示
        pooled_output = torch.cat(head_outputs, dim=-1)  # 输出的维度是 input_dim

        output = self.linear(pooled_output)
        
        return output
  

# 定义一个两层投射层
class Linear(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=4096):
        super(Linear, self).__init__()
        self.layer1 = nn.Linear(input_dim, 2048)  # 第一层线性层
        self.relu = nn.ReLU()  # ReLU 激活层
        self.layer2 = nn.Linear(2048, hidden_dim)  # 第二层线性层

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

##重定义mask
def adjust_attention_mask(combin_attn_mask, content_start_end_positions, audio_start_list, video_start_list):
    len_seq = combin_attn_mask.size(0)
    # 将原始的一维掩码扩展成二维矩阵
    extended_attn_mask = combin_attn_mask.unsqueeze(0).repeat(len_seq, 1)

    # 音频和视频特征长度固定为32
    feature_length = 32
    content_start, content_end = content_start_end_positions
    # 处理音频特征与内容区域不交互
    for audio_start in audio_start_list:
        audio_end = audio_start + feature_length
        extended_attn_mask[content_start:content_end, audio_start:audio_end] = 0
        extended_attn_mask[audio_start:audio_end, content_start:content_end] = 0

    # 处理视频特征与内容区域不交互
    for video_start in video_start_list:
        video_end = video_start + feature_length
        extended_attn_mask[content_start:content_end, video_start:video_end] = 0
        extended_attn_mask[video_start:video_end, content_start:content_end] = 0

    return extended_attn_mask #dtype=attn_mask.dtype


# 定义多模态模型
class MultiLlmModel(nn.Module):
    def __init__(self, llama_model, tokenizer, LQ_wav_model, LQ_vid_model):
        super(MultiLlmModel, self).__init__()

        if args.dataset == "CMDC":
            self.video_embedding_dim = 709+1
            self.audio_embedding_dim = 768
        else:
            self.video_embedding_dim = 1024#1024
            self.audio_embedding_dim = 1024
           
        self.llama_model = llama_model
        self.tokenizer = tokenizer
        
        if args.LQ_former or args.GD_llm:
            self.LQ_wav_model = LQ_wav_model
            self.LQ_vid_model = LQ_vid_model
            self.vid_projection = Linear(input_dim=self.video_embedding_dim)
            self.wav_projection = Linear(input_dim=self.audio_embedding_dim)
            self.wav_tgt = nn.Parameter(torch.randn(32, self.audio_embedding_dim))
            self.vid_tgt = nn.Parameter(torch.randn(32, self.video_embedding_dim))
            if args.dataset == "E-DAIC-WOZ":
                self.pool = nn.MaxPool1d(kernel_size=3, stride=3)#不包含可学习的参数
                self.linear_video = nn.Linear(2048, self.video_embedding_dim)

        if use_class:
            self.MF = AttentionPooling(input_dim=4096)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, sample):
        device = self.device

        # 将所有输入张量移至同一设备
        texts_ids = sample['input_ids'].to(device)
        attention_mask = sample['attention_mask'].to(device)
        insert_audio_positions = sample['insert_audio_positions']
        insert_video_positions = sample['insert_video_positions']
        audio_embeddings = sample['audio_embeddings'].to(device)  # 移动到 GPU
        audio_att_mask = sample['audio_att_mask'].to(device)
        video_embeddings = sample['video_embeddings'].to(device)
        video_att_mask = sample['video_att_mask'].to(device)
        labels_ids = sample['labels'].to(device)
        label = torch.tensor(sample['label'], dtype=torch.float32).to(device)

        content_start_end_positions = sample['content_start_end_positions']

        # print("content_start_end_positions",content_start_end_positions)

        if torch.all(labels_ids == -100):####labels_ids全为-100
            print("labels_ids contains only -100 values")

        batch_size = texts_ids.size(0)

        if args.LQ_former or args.GD_llm:
            wav_query = self.wav_tgt.unsqueeze(0).expand(batch_size, -1, -1).to(device)  #########可学习参数
            vid_query = self.vid_tgt.unsqueeze(0).expand(batch_size, -1, -1).to(device)  #########可学习参数

            if args.dataset == "E-DAIC-WOZ":
                audio_embeddings = self.pool(audio_embeddings.transpose(1, 2)).transpose(1, 2)
                audio_att_mask = self.pool(audio_att_mask.unsqueeze(1).float()).squeeze(1)

                video_embeddings = self.linear_video(video_embeddings)
                
            audio_features = self.wav_projection(self.LQ_wav_model(wav_query, audio_embeddings, audio_att_mask))
            video_features = self.vid_projection(self.LQ_vid_model(vid_query, video_embeddings, video_att_mask))

            embs, attn_masks, adjusted_labels = self.get_context_emb(
                texts_ids, attention_mask, insert_audio_positions,
                insert_video_positions, audio_features, video_features, content_start_end_positions, labels_ids
            )
    
            assert embs.size(1) == adjusted_labels.size(1), \
                f"Embeddings length {embs.size(1)} and labels length {adjusted_labels.size(1)} do not match."
        else:
            embs = self.embed_tokens(texts_ids)  # Get text embeddings
            attn_masks = attention_mask
            adjusted_labels = labels_ids

        
        with self.maybe_autocast():
            output = self.llama_model(
                inputs_embeds=embs,
                attention_mask=attn_masks,
                labels=adjusted_labels,
                output_hidden_states=True,
            )

        if use_class:
            category_output = self.MF(output.hidden_states[-1])
            #####在这里补充获取生成部分的token
            print('预测值与真值：',category_output.squeeze(1), label)
            category_loss = criterion(category_output.squeeze(1), label)
    
            return output, category_loss
        else:
             return output, None
    
    # 根据 insert_audio_positions, insert_video_positions 插入其他模态特征
    def get_context_emb(self, texts_ids, attns_mask, insert_audio_positions, insert_video_positions, embs_audios, embs_video, content_start_end_positions, labels_ids=None):
        device = self.device

        all_embs = []
        all_attn_masks = []
        adjusted_labels_list = []

        #遍历batch处理每个样本
        for i in range(texts_ids.size(0)):
            text_ids = texts_ids[i]
            emb = self.embed_tokens(text_ids)  # Get text embeddings
            attn_mask = attns_mask[i]

            if labels_ids is not None:
                labels = labels_ids[i]

            # 获取需要插入音频特征的位置
            positions_wav = insert_audio_positions[i]
            positions_vid = insert_video_positions[i]
            # 包含了处理多个音频和视频的情况，如positions_wav=[[],[]]
            positions = [0] + positions_wav + positions_vid + [emb.size(0)]  # 确保起始和结束位置

            # 分割文本嵌入和注意力掩码
            embs_text = [emb[positions[j]:positions[j + 1], :] for j in range(len(positions) - 1)]##################2个位置就分割为3段
            attn_masks_split = [attn_mask[positions[j]:positions[j + 1]] for j in range(len(positions) - 1)]

            if labels_ids is not None:
                labels_split = [labels[positions[j]:positions[j + 1]] for j in range(len(positions) - 1)]

            if len(embs_text) != 3:
                raise ValueError(f"Unexpected length of embs_text: {len(embs_text)}. Content: {embs_text}")
            
            combin_embs = torch.cat((embs_text[0], embs_audios[i], embs_text[1], embs_video[i], embs_text[2]), dim=0)
            
            combin_attn_masks = torch.cat((attn_masks_split[0], torch.ones(embs_audios[i].size(0), dtype=attn_mask.dtype).to(device), attn_masks_split[1], torch.ones(embs_video[i].size(0), dtype=attn_mask.dtype).to(device), attn_masks_split[2]), dim=0)

            # 对combin_attn_masks再定义
            # 根据 positions_wav、positions_vid 获取插入的语音和视觉位置，要考虑插入的特征长度（一般len=32）
            # 根据content_start_end_positions[i]（格式[tenser(12),tenser(55)]）获取content内容区域
            # 希望补全这部分代码，1.希望对combin_attn_masks重新赋值。2.插入的语音和视觉特征不与content位置的信息进行交互，但是与其他位置信息交互。3.其他位置的原交互逻辑不变。
            # 4.先将combin_attn_masks由2维扩展成3维度，然后处理上述逻辑，返回（B,len,len）格式。
            # combin_attn_masks = adjust_attention_mask(combin_attn_masks, content_start_end_positions, positions_wav, positions_vid)

            if labels_ids is not None:
                adjusted_labels = torch.cat((labels_split[0], torch.full((embs_audios[i].size(0),), -100, dtype=labels.dtype).to(device), labels_split[1], torch.full((embs_video[i].size(0),), -100, dtype=labels.dtype).to(device), labels_split[2]), dim=0)

            all_embs.append(combin_embs)
            all_attn_masks.append(combin_attn_masks)#变3维
            if labels_ids is not None:
                adjusted_labels_list.append(adjusted_labels)
        
        padded_embs = torch.stack(all_embs)
        padded_attn_masks = torch.stack(all_attn_masks)#[B,len,len]

        if labels_ids is not None:
            padded_labels = torch.stack(adjusted_labels_list)
 
        if labels_ids is not None:
            return padded_embs, padded_attn_masks, padded_labels
        else:
            return padded_embs, padded_attn_masks


    def embed_tokens(self, token_ids):
        if hasattr(self.llama_model.base_model, 'model'):  # lora wrapped model
            embeds = self.llama_model.base_model.model.model.embed_tokens(token_ids)
        else:
            embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds

    def maybe_autocast(self, dtype=torch.float16):
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.amp.autocast(device_type='cuda', dtype=dtype)
        else:
            return contextlib.nullcontext()

    def generate(self, sample, max_new_tokens=50):
        device = self.device
        texts_ids = sample['input_ids'].to(device)
        attention_mask = sample['attention_mask'].to(device)
        insert_audio_positions = sample['insert_audio_positions']
        insert_video_positions = sample['insert_video_positions']
        audio_embeddings = sample['audio_embeddings'].to(device)  # 移动到 GPU
        audio_att_mask = sample['audio_att_mask'].to(device)
        video_embeddings = sample['video_embeddings'].to(device)
        video_att_mask = sample['video_att_mask'].to(device)

        content_start_end_positions = sample['content_start_end_positions']

        batch_size = texts_ids.size(0)

        if args.LQ_former or args.GD_llm:
            wav_query = self.wav_tgt.unsqueeze(0).expand(batch_size, -1, -1).to(device)  #########可学习参数
            vid_query = self.vid_tgt.unsqueeze(0).expand(batch_size, -1, -1).to(device)  #########可学习参数
            
            if args.dataset == "E-DAIC-WOZ":
                
                audio_embeddings = self.pool(audio_embeddings.transpose(1, 2)).transpose(1, 2)
                audio_att_mask = self.pool(audio_att_mask.unsqueeze(1).float()).squeeze(1)

                video_embeddings = self.linear_video(video_embeddings)
                
            audio_features = self.wav_projection(self.LQ_wav_model(wav_query, audio_embeddings, audio_att_mask))
            video_features = self.vid_projection(self.LQ_vid_model(vid_query, video_embeddings, video_att_mask))

            # 获取调整后的嵌入和注意力掩码
            embs, attn_masks = self.get_context_emb(
                texts_ids, attention_mask, insert_audio_positions,
                insert_video_positions, audio_features, video_features, content_start_end_positions, labels_ids=None,
            )
        else:
            embs = self.embed_tokens(texts_ids)  # Get text embeddings
            attn_masks = attention_mask

        # 使用 generate 方法生成文本
        with self.maybe_autocast():
            generated_ids = self.llama_model.generate(
                inputs_embeds=embs,
                attention_mask=attn_masks,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_k=10,  # 采样策略，它限制了模型在每一步生成时仅从概率最高的前 50 个词汇中进行选择。
                pad_token_id = self.tokenizer.pad_token_id,
                # eos_token_id = self.tokenizer.eos_token_id,
                # repetition_penalty=1.1, # 惩罚重复内容
            )

        
        if use_class:
            # 使用 forward 方法生成文本并获取 hidden_states
            with self.maybe_autocast():
                output = self.llama_model(
                    inputs_embeds=embs,
                    attention_mask=attn_masks,
                    max_length=max_new_tokens,
                    return_dict=True,  # 返回字典形式的输出
                    output_hidden_states=True  # 返回所有的 hidden states
                )

            category_label = self.MF(output.hidden_states[-1])
            return generated_ids, category_label.squeeze(1)
        else:
            return generated_ids, None




if __name__ == "__main__":
    pass