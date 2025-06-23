import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader
import os
import json
from torch.nn.utils.rnn import pad_sequence
from model.decoder import Transformer_decoder
import time
import pandas as pd
import numpy as np
import scipy.io

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = "E-DAIC-WOZ"#""E-DAIC-WOZ"#"CMDC"#
mode = "audio"#audio   video

if dataset == "CMDC":
    pkl_dir = 'D:/Desktop/CMDC/'
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
    pkl_dir = 'D:/Desktop'
    def load_video_embedding(mat_file_path):
        pkl_file_path = mat_file_path.replace('\\', '/')
        # 构建完整路径
        full_pkl_file_path = os.path.join(pkl_dir, pkl_file_path)
        # 读取 .mat 文件
        mat_data = scipy.io.loadmat(full_pkl_file_path)['feature']
        # 打印文件中的内容
        # 抽帧操作
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
        if dataset == 'CMDC':
            return audio_embedding[:2048]
        else:
            return audio_embedding[:21000]
    else:
        print(f"Error: {full_pkl_file_path} does not exist.")
        return None


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
    padded_embeddings = pad_sequence(embeddings_list, batch_first=True, padding_value=0).to(device)
    att_mask = pad_sequence(att_masks, batch_first=True, padding_value=0).to(device, dtype=torch.float32)

    # print(padded_embeddings.shape, tyatt_mask.shape)
    return padded_embeddings, att_mask


# 创建数据集类
class MultiQnADataset(Dataset):
    def __init__(self, data, audio_embedding_dim=768, video_embedding_dim=709):
        self.data = data
        self.video_embedding_dim = video_embedding_dim +1
        self.audio_embedding_dim = audio_embedding_dim

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):

        key, items = self.data[idx]
        label = items['label']
        if mode == "video":
            video_list_paths = items.get('video_embedding_path', [])
            embeddings = process_embedding(video_list_paths, load_video_embedding,self.video_embedding_dim)
        else:
            audio_list_paths = items.get('audio_embedding_pkl_path', [])
            embeddings = process_embedding(audio_list_paths,load_audio_embedding,self.audio_embedding_dim)

        return {
            f'{mode}_embedding': embeddings.to(device),
            # 'video_embedding': video_embeddings.to(device),
            'label':label,
        }

    def collate_fn(self, batch):

        # 处理 audio_embedding
        embeddings_list = []
        label_list = []
        for item in batch:
            # 检查 audio_embedding 是否全零
            if not torch.all(item[f'{mode}_embedding'] == 0):
                embeddings_list.append(item[f'{mode}_embedding'])
                label_list.append(item['label'])
        # print('检查 audio_embedding 是否全零', time.time() - s_time)
        # 对筛选后的音频嵌入列表进行填充
        if embeddings_list:
            # s_time = time.time()
            embeddings, att_mask = process_padding(embeddings_list)
            # print('process_padding',time.time()-s_time)
            label = torch.tensor(label_list).to(device)
        else:
            return None

        return {
            'label': label,
            f'{mode}_embedding':(embeddings,att_mask),
        }


# 定义一个简单的线性分类器
class LinearClassifier(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim = 512, num_classes=4):
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 线性层
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)  # 第一层线性变换
        x = self.relu(x)  # 激活函数
        x = self.fc2(x)  # 第二层线性变换
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        nhead = 8
        if dataset == "CMDC":
            if mode == "video":
                self.input_size = 709+1
                nhead=5
            else:
                self.input_size = 768 # 要与MultiQnADataset dim 一致
        else:
            self.input_size = 1024#1024

        self.hidden_size = 1024

        self.former = Transformer_decoder(d_model=self.input_size, nhead=nhead, num_decoder_layers=4, dim_feedforward=1024, dropout=0.3)
        self.tgt = nn.Parameter(torch.randn(32, self.input_size))
        self.linear = LinearClassifier(input_dim=self.input_size, num_classes=1)###Transformer的输出是input_size

        if dataset == "E-DAIC-WOZ":
            self.pool = nn.MaxPool1d(kernel_size=3, stride=3)  # 不包含可学习的参数
            self.linear_video = nn.Linear(2048, self.input_size)

    def forward(self, anchor_tuple):
        anchor, att_mask = anchor_tuple
        batch_size, seq_anchor_len, feature_dim = anchor.shape
        # print(batch_size, seq_anchor_len, feature_dim)
        query = self.tgt.unsqueeze(0).expand(batch_size, -1, -1)#########可学习参数
        if seq_anchor_len <= 2:
            anchor_encoded = torch.zeros(batch_size, 1, self.hidden_size)
        else:
            if dataset == "E-DAIC-WOZ":#E-DAIC-WOZ":
                if mode == "video":
                    anchor = self.linear_video(anchor)
                else:
                    anchor = self.pool(anchor.transpose(1, 2)).transpose(1, 2)
                    att_mask = self.pool(att_mask.unsqueeze(1).float()).squeeze(1)
                    # print(att_mask[0,:])
            # print(anchor.shape, att_mask.shape, query.shape)
            anchor_encoded = self.former(query, anchor, att_mask)
            anchor_encoded = anchor_encoded.mean(dim=1)  ##均值聚合
        # print('anchor_encoded', anchor_encoded.shape)
        output = self.linear(anchor_encoded)
        return output

if __name__ == "__main__":
    save_path = 'D:/Desktop/MLlm-DR/checkpoint'
    # 准备数据集和数据加载器
    if dataset == "CMDC":
        with open('D:/Desktop/CMDC/output_data02.json', 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    else:
        with open('D:/Desktop/E-DAIC-WOZ/output02.json', 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

    if dataset == "CMDC":
        # 划分训练集、测试集
        def split_data(train_data, test_data):
            for key, value in raw_data.items():
                if key.startswith("HC"):
                    # 提取数字部分
                    num_str = key[2:].split('-')[0]   # 从索引2开始到字符串结束的部分
                    # 将字符串转换为整数
                    num = int(num_str)
                    if 0 < num < 41:
                        train_data.append((key, value))
                    else:
                        test_data.append((key, value))
                else:
                    # 提取数字部分
                    num_str = key[3:].split('-')[0]   # 从索引2开始到字符串结束的部分
                    # 将字符串转换为整数
                    num = int(num_str)
                    if 0 < num < 21:
                        train_data.append((key, value))
                    else:
                        test_data.append((key, value))
            return train_data, test_data
    else:
        def split_data(train_data, test_data):
            train_split = pd.read_csv('D:/Desktop/E-DAIC-WOZ/train_split.csv')
            dev_split = pd.read_csv('D:/Desktop/E-DAIC-WOZ/dev_split.csv')
            test_split = pd.read_csv('D:/Desktop/E-DAIC-WOZ/test_split.csv')

            # 提取 Participant_ID 列
            train_participants = train_split['Participant_ID'].tolist()
            dev_participants = dev_split['Participant_ID'].tolist()
            test_participants = test_split['Participant_ID'].tolist()

            for key, value in raw_data.items():
                key_name = int(key.split('-')[0])
                if key_name in train_participants:
                    train_data.append((key, value))
                elif key_name in test_participants:
                    test_data.append((key, value))

            return train_data, test_data

    def to_device(tensor_tuple, device):
        return tuple(tensor.to(device) for tensor in tensor_tuple)

    train_data, test_data = split_data(train_data=[], test_data=[])
    print(len(train_data),len(test_data))

    # 初始化数据集
    train_dataset = MultiQnADataset(train_data)
    test_dataset = MultiQnADataset(test_data)

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=train_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=test_dataset.collate_fn)

    model = Model()

    import torch.optim as optim
    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # 训练和验证
    num_epochs = 50
    model.to(device)

    # 损失函数更改为均方误差损失
    criterion = nn.MSELoss()

    # 预训练模型的路径
    pretrained_model_path = f'D:/Desktop/LLaMA-DR/LLaMA-DR/checkpoint/{dataset}_{mode}_best_model.pth'

    # 加载预训练模型
    if os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path))
        print(f"Loaded pretrained model from {pretrained_model_path}")
    else:
        print("No pretrained model found. Training from scratch.")

    # 初始化最小验证损失值
    best_val = 2.0

    # 开始训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct_preds = 0.0
        train_total_preds = 0.0
        for batch_idx, batch_data in enumerate(train_loader):##数据加载消耗大量时间
            if batch_data is not None:
                anchor = to_device(batch_data[f'{mode}_embedding'], device)
                label = batch_data['label'].to(device).float()
                optimizer.zero_grad()
                output = model(anchor)
                # print(output.squeeze(),label)
                loss = criterion(output.squeeze(),label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # 计算回归任务的评估指标（例如均方误差）
                train_correct_preds += (output.squeeze() - label).abs().sum().item()
                train_total_preds += label.size(0)

        train_avg_mae = train_correct_preds / train_total_preds
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}, MAE: {train_avg_mae:.4f}")

        # 验证
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                test_correct_preds = 0
                test_total_preds = 0
                for batch_idx, batch_data in enumerate(test_loader):
                    if batch_data is not None:
                        anchor = to_device(batch_data[f'{mode}_embedding'], device)
                        label = batch_data['label'].to(device).float()

                        output = model(anchor)
                        loss = criterion(output.squeeze(), label)
                        test_loss += loss.item()

                        # 计算回归任务的评估指标（例如均方误差）
                        test_correct_preds += ((output.squeeze() - label).abs().sum().item()) /label.size(0)

                print(test_loss, test_correct_preds)
                avg_test_loss = test_loss / len(test_loader)
                # 计算回归任务的平均绝对误差 (MAE)
                avg_mae = test_correct_preds / len(test_loader)

                print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_test_loss:.4f}, MAE: {avg_mae:.4f}")

            # 保存模型参数
            if avg_test_loss < best_val:
                best_val = avg_test_loss
                best_model_path = os.path.join(save_path, f'{dataset}_{mode}_best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with Val loss: {best_val:.4f}")

    best_model_path = os.path.join(save_path, f'end{num_epochs}_{dataset}_{mode}_model.pth')
    torch.save(model.state_dict(), best_model_path)