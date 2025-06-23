import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
import re
import os
import pickle
import os
import pandas as pd
import numpy as np
import scipy.io
from opts import parse_opts
args = parse_opts()


if args.dataset == 'CMDC':
    pkl_dir = '/root/autodl-tmp/CMDC/'
    def load_video_embedding(csv_file_path):
        # 确保文件路径非空
        if not csv_file_path:
            return None
        # 将路径中的反斜杠替换为正斜杠并构建完整路径
        csv_file_path = csv_file_path.replace('\\', '/')
        full_csv_file_path = os.path.join(pkl_dir, csv_file_path)
    
        # 检查 CSV 文件是否存在
        if os.path.isfile(full_csv_file_path):
            # 读取 CSV 文件并去除不需要的列
            video_embedding = pd.read_csv(full_csv_file_path)
            # 去掉指定列
            columns_to_drop = ['frame', ' face_id', ' timestamp', ' confidence', ' success']
            video_embedding = video_embedding.drop(columns=columns_to_drop, errors='ignore').values
            # 检查特征维度是否为 709
            if video_embedding.shape[1] != 709:
                print(f"Error: {full_csv_file_path} has incorrect feature dimension: {video_embedding.shape[1]}")
    
            # 补齐特征向量到 710 维  如果是使用transformer
            if video_embedding.shape[1] == 709:
                video_embedding = np.pad(video_embedding, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    
            # 只取前 4196 行
            return video_embedding[:2048]
        else:
            print(f"Error: {full_csv_file_path} does not exist.")
            return None
else:
    pkl_dir = '/root/autodl-tmp/'
    def load_video_embedding(mat_file_path):
        mat_file_path = mat_file_path.replace('\\', '/')
        # 构建完整路径
        full_pkl_file_path = os.path.join(pkl_dir, mat_file_path)
        # 读取 .mat 文件
        mat_data = scipy.io.loadmat(full_pkl_file_path)['feature']
        #抽帧操作
        step = 3
        return mat_data[:21000:step]


def load_audio_embedding(pkl_file_path):
    # 检查 pkl_file_path 是否为空
    if not pkl_file_path:
        return None
    # 将路径中的反斜杠替换为正斜杠
    pkl_file_path = pkl_file_path.replace('\\', '/')
    # 构建完整路径
    full_pkl_file_path = os.path.join(pkl_dir, pkl_file_path)
    # 检查 pkl 文件是否存在
    if os.path.isfile(full_pkl_file_path):
        # 载入音频嵌入
        with open(full_pkl_file_path, 'rb') as f:
            audio_embedding = pickle.load(f)
        if args.dataset == 'CMDC':
            return audio_embedding[:2048]
        else:
            return audio_embedding[:21000]
    else:
        print(f"Error: {full_pkl_file_path} does not exist.")
        return None


def find_all_occurrences_re(text, substring):
    return [match.start() for match in re.finditer(re.escape(substring), text)]

i = 0
def apply_template(messages, tokenizer, max_tokens=2048):
    global i 
    
    if args.LQ_former or args.GD_llm:
        text = f"<AudioFeature>\n\n<AudioHere><|eot_id|></AudioFeature><VideoFeature>\n\n<VideoHere><|eot_id|></VideoFeature>"
    else:
        text = ""
    for idx, msg in enumerate(messages):
        role = msg['role']
        content = msg['content'].strip()
        if role == 'user':
            # 使用 encode 返回 token ID
            content_tokens = tokenizer.encode(content)#########################第一个内容长度不超过100
            # print('内容长度：', len(content_tokens))
            if len(content_tokens) > max_tokens:
                # 截断到指定长度
                content = tokenizer.decode(content_tokens[:max_tokens])
                # print('截断后的内容：', content)
        # 根据角色生成模板
        if role == 'system':
            text += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == 'user':
            text += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == 'assistant':
            text += f"<|start_header_id|>assistant<|end_header_id|>\n\n<ASSISTANT>{content}<|eot_id|></ASSISTANT>"

    # print('整体模板长度：',len(tokenizer.tokenize(text)))
    if i ==0:
        print(text)
        i = i+1
    return text

def process_embedding(list_paths, load_func, dim):
    # 没有文本时，list_paths可能为空
    all_embeddings = []
    for path in list_paths:
        embedding = load_func(path)
        if embedding is None:
            embedding = torch.zeros(1, dim, dtype=torch.float32)  # 指定 float32 类型
        else:
            embedding = torch.tensor(embedding, dtype=torch.float32)  # 转换为 float32
        all_embeddings.append(embedding)

    if all_embeddings:
        embeddings = torch.cat(all_embeddings, dim=0)  # 序列维度拼接
    else:
        embeddings = torch.zeros(1, dim, dtype=torch.float32)  # 如果没有音频嵌入
    return embeddings


def process_padding(embeddings_list):
    # Create attention masks with 1 for real values and 0 for padding
    att_masks = [torch.ones(embedding.size(0), dtype=torch.long) for embedding in embeddings_list]

    # Pad the embeddings and the attention masks
    padded_embeddings = pad_sequence(embeddings_list, batch_first=True, padding_value=0)
    att_mask = pad_sequence(att_masks, batch_first=True, padding_value=0).to(dtype=torch.float32)

    # print(padded_embeddings.shape, tyatt_mask.shape)
    return padded_embeddings, att_mask


# 创建数据集类
class MultiQnADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=3072, load_test=False):
        if args.dataset == 'CMDC':
            self.audio_embedding_dim = 768
            self.video_embedding_dim = 709+1
        else:
            self.audio_embedding_dim = 1024
            self.video_embedding_dim = 2048
            
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length########允许的最长输入
        self.load_test = load_test
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key, items = self.data[idx]
        messages = items['messages']
        label = items['label']

        if self.load_test:
             messages = [msg for msg in messages if msg['role'] != 'assistant']  # 移除 assistant 的回复
            
        # 模板处理
        inputs_text = apply_template(messages,self.tokenizer, self.max_length-300)

        # Tokenization
        tokenized_inputs = self.tokenizer(
            inputs_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            # padding=True, #在collate_fn再padding
        )

        input_ids = tokenized_inputs['input_ids'].squeeze(0)
        attention_mask = tokenized_inputs['attention_mask'].squeeze(0)

        # 获取特殊标记的 Token ID
        audio_here_token_id = self.tokenizer.convert_tokens_to_ids('<AudioHere>')
        video_here_token_id = self.tokenizer.convert_tokens_to_ids('<VideoHere>')

        # 在 input_ids 中找到特殊标记的位置
        insert_audio_positions = (input_ids == audio_here_token_id).nonzero(as_tuple=True)[0].tolist()
        insert_video_positions = (input_ids == video_here_token_id).nonzero(as_tuple=True)[0].tolist()

        # print(insert_audio_positions, insert_video_positions)

        # 移除特殊标记的 token，以避免后续处理中的干扰
        indices_to_remove = (input_ids == audio_here_token_id) | (input_ids == video_here_token_id)
        input_ids = input_ids[~indices_to_remove]
        attention_mask = attention_mask[~indices_to_remove]

        # 处理音频嵌入
        audio_list_paths = items.get('audio_embedding_pkl_path', [])
        audio_embeddings = process_embedding(audio_list_paths,load_audio_embedding,self.audio_embedding_dim)

        video_list_paths = items.get('video_embedding_path', [])
        video_embeddings = process_embedding(video_list_paths, load_video_embedding,self.video_embedding_dim)

        content_start_token = self.tokenizer.convert_tokens_to_ids('<|end_header_id|>')
        content_end_token = self.tokenizer.convert_tokens_to_ids('<|eot_id|>')
        content_start_positions = (input_ids == content_start_token).nonzero(as_tuple=True)[0]
        content_end_positions = (input_ids == content_end_token).nonzero(as_tuple=True)[0]

        if self.load_test:
            content_start_positions = content_start_positions[1]  # 顺数第二个位置
            content_end_positions = content_end_positions[-1]  # 倒是第二个位置

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'insert_audio_position': insert_audio_positions,
                'insert_video_position': insert_video_positions,
                'content_start_end_positions': [content_start_positions, content_end_positions],
                'audio_embedding': audio_embeddings,
                'video_embedding': video_embeddings,
                'key': key,  # 添加样本的键值，方便调试
                }
        else:
            # 检查eos_token_id的位置
            # eos_token_id = self.tokenizer.eos_token_id
            # eos_indices = (input_ids == eos_token_id).nonzero(as_tuple=True)[0]
            # print(f"EOS token indices in input_ids: {eos_indices}")

            content_start_positions = content_start_positions[1]# 顺数第二个位置
            content_end_positions = content_end_positions[-2]# 倒是第二个位置


            # 创建与 input_ids 等长的 labels 张量，初始值为 -100
            labels = torch.full_like(input_ids, -100)
    
            # 处理 <ASSISTANT> 标签的部分
            assistant_start_token_id = self.tokenizer.convert_tokens_to_ids('<ASSISTANT>')
            assistant_end_token_id = self.tokenizer.convert_tokens_to_ids('</ASSISTANT>')

            assistant_start_indices = (input_ids == assistant_start_token_id).nonzero(as_tuple=True)[0]
            assistant_end_indices = (input_ids == assistant_end_token_id).nonzero(as_tuple=True)[0]

            # print('lable标记位置：',assistant_start_indices,assistant_end_indices)
    
            if len(assistant_start_indices) != len(assistant_end_indices):
                print(f"Error in sample {key}: Mismatch in assistant tokens.")
                print(f"Assistant start indices: {assistant_start_indices}")
                print(f"Assistant end indices: {assistant_end_indices}")
                raise ValueError("Mismatch between number of assistant start and end tokens.")
    
            for start_idx, end_idx in zip(assistant_start_indices, assistant_end_indices):
                # 将助手回复部分的标签设置为对应的 token id
                labels[start_idx + 1:end_idx] = input_ids[start_idx + 1:end_idx]
    
            # 将特殊标记的位置的标签设为 -100
            labels[input_ids == assistant_start_token_id] = -100
            labels[input_ids == assistant_end_token_id] = -100
    
            # 检查labels中eos_token_id的位置值
            # eos_labels = labels[eos_indices]
            # print(f"Values at EOS token indices in labels: {eos_labels}")

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'insert_audio_position': insert_audio_positions,
                'insert_video_position': insert_video_positions,
                'content_start_end_positions':[content_start_positions,content_end_positions],
                'audio_embedding': audio_embeddings,
                'video_embedding': video_embeddings,
                'labels': labels, #是生成的句子真值
                'key': key,  # 添加样本的键值，方便调试
                'label': label, #是句子级别的类别标签
            }

    def collate_fn(self, batch):
        # 处理 input_ids 和 attention_mask
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]

        # 使用 tokenizer 的 pad 方法在 collate_fn 中对整个批次进行填充
        tokenized_batch = self.tokenizer.pad(
            {'input_ids': input_ids, 'attention_mask': attention_masks},
            padding=True,
            return_tensors="pt"
        )

        if self.load_test:
            pass
        else:
            labels = [item['labels'] for item in batch]
            # 对 labels 进行手动填充
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        # 处理 audio_embedding
        audio_embeddings_list = [item['audio_embedding'] for item in batch]
        audio_embeddings, audio_att_mask = process_padding(audio_embeddings_list)

        video_embeddings_list = [item['video_embedding'] for item in batch]
        video_embeddings, video_att_mask = process_padding(video_embeddings_list)

        # 处理 insert positions
        insert_audio_positions = [item['insert_audio_position'] for item in batch]
        insert_video_positions = [item['insert_video_position'] for item in batch]

        content_start_end_positions = [item['content_start_end_positions'] for item in batch]

        if self.load_test:
             return {
                'input_ids': tokenized_batch['input_ids'],
                'attention_mask': tokenized_batch['attention_mask'],
                'insert_audio_positions': insert_audio_positions,
                'insert_video_positions': insert_video_positions,
                'content_start_end_positions': content_start_end_positions,
                'audio_embeddings': audio_embeddings,
                'audio_att_mask': audio_att_mask,
                'video_embeddings': video_embeddings,
                'video_att_mask': video_att_mask,
                'keys': [item['key'] for item in batch],  # 添加样本的键值，方便调试
            }
        else:
            return {
                'input_ids': tokenized_batch['input_ids'],
                'attention_mask': tokenized_batch['attention_mask'],
                'insert_audio_positions': insert_audio_positions,
                'insert_video_positions': insert_video_positions,
                'content_start_end_positions': content_start_end_positions,
                'audio_embeddings': audio_embeddings,
                'audio_att_mask': audio_att_mask,
                'video_embeddings': video_embeddings,
                'video_att_mask': video_att_mask,
                'labels': labels,
                'keys': [item['key'] for item in batch],  # 添加样本的键值，方便调试
                'label': [item['label'] for item in batch],
            }


if __name__ == "__main__":

    raw_data ={
            "HC01-1":{
            "messages":[
                {'role': 'system', 'content': "你是一名精神科医生，通过提问了解参与者在抑郁相关的某些方面。0表示完全没有，1表示有几天，2表示超过一半时间，3表示几乎每天。"},
                {'role': 'user', 'content': "问题4：最近你和朋友的交流频率如何？你最好的朋友如何评价你？回答4：可以说是天天都交流吧，有时候会吐槽一下，实验室，几乎天天都会说一说。"},
                {'role': 'user','content': "问题6：你对目前的学习或工作感兴趣吗？你在日常生活中的注意力如何？回答6：专注的话，感觉一般吧，因为我学习的话。基本上是坐在那里，可以学认真的学五十分钟，但是再往后面学就感觉很吃力，所以我就会休息休息一段时间，然后再去学，就是差不多极限就是五十分钟了。"},
                {'role': 'user', 'content': "请根据对话内容给出参与者在对做事情失去了兴趣或乐趣方面的评估结果和评估理由。要求：评估结果为0-3的数值，评估理由尽量简洁。"},
                {'role': "assistant", 'content': "评估结果：1。评估理由：患者依然保持与朋友的频繁交流，并提到偶尔吐槽实验室生活，表明其在社交活动中仍然能够获得一定的乐趣。同时，他在学习时能够持续集中注意力约50分钟，尽管之后感到吃力，需要休息，但这显示出他对学习仍有一定的兴趣和投入。因此，患者在日常生活中并未完全对活动失去兴趣，但可能存在轻微的兴趣下降。"}],
            "audio_embedding_pkl_path": ["D:/Desktop/CMDC/HC01-1/Q6_hubert_wav.pkl",
                                         "D:/Desktop/CMDC/HC01-1/Q6_hubert_wav.pkl"],
            "label": 1
    },

            "HC01-2":{
            "messages":[
                {'role': 'system', 'content': "你是一名精神科医生，通过提问了解参与者在抑郁相关的某些方面。0表示完全没有，1表示有几天，2表示超过一半时间，3表示几乎每天。"},
                {'role': 'user', 'content': "问题4：最近你和朋友的交流频率如何？你最好的朋友如何评价你？回答4：可以说是天天都交流吧，有时候会吐槽一下。"},
                {'role': 'user','content': "问题6：你对目前的学习或工作感兴趣吗？你在日常生活中的注意力如何？回答6：专注的话，感觉一般吧，因为我学习的话。基本上是坐在那里，可以学认真的学五十分钟，但是再往后面学就感觉很吃力，所以我就会休息休息一段时间，然后再去学，就是差不多极限就是五十分钟了。"},
                {'role': 'user', 'content': "请根据对话内容给出参与者在对做事情失去了兴趣或乐趣方面的评估结果和评估理由。要求：评估结果为0-3的数值，评估理由尽量简洁。"},
                {'role': "assistant", 'content': "评估结果：1。评估理由：患者依然保持与朋友的频繁交流，并提到偶尔吐槽实验室生活，表明其在社交活动中仍然能够获得一定的乐趣。同时，他在学习时能够持续集中注意力约50分钟，尽管之后感到吃力，需要休息，但这显示出他对学习仍有一定的兴趣和投入。因此，患者在日常生活中并未完全对活动失去兴趣，但可能存在轻微的兴趣下降。"}],
           "audio_embedding_pkl_path": ["D:/Desktop/CMDC/HC01-1/Q6_hubert_wav.pkl",
                                         "D:/Desktop/CMDC/HC01-1/Q6_hubert_wav.pkl"],
            "label": 1
            }
        }

    data = [(key, value) for key, value in raw_data.items()]


    def setup_tokenizer(model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokenizer.pad_token = tokenizer.eos_token

        special_tokens = {
            'additional_special_tokens': [
                '<AudioFeature>', '</AudioFeature>',
                '<VideoFeature>', '</VideoFeature>',
                '<AudioHere>', '<VideoHere>',
                '<ASSISTANT>', '</ASSISTANT>',
                # '<|begin_of_text|>', '<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>'
            ]
        }
        tokenizer.add_special_tokens(special_tokens)
        return tokenizer


    model_dir = 'D:\Desktop\MLlm-DR\llama-3-chinese-8b-instruct-v3'
    tokenizer = setup_tokenizer(model_dir)

    # 初始化数据集
    full_dataset = MultiQnADataset(data,tokenizer)
    train_loader = DataLoader(full_dataset, batch_size=2, shuffle=True, collate_fn=full_dataset.collate_fn)
    for batch_idx, batch_data in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}:")
        print(batch_data['input_ids'].size())
        print(batch_data['attention_mask'].size())
        print(batch_data['insert_audio_positions'])
        print(batch_data['content_start_end_positions'])
        print(batch_data['audio_embeddings'].size())#([2, 2, 1061, 768])
        print(batch_data['labels'].size())
