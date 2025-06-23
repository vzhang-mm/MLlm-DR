#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import json
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from dataset.dataset import MultiQnADataset, apply_template, load_audio_embedding, process_embedding, load_video_embedding
# from model.cnn import PyramidPooling1D as CNN
from model.decoder import Transformer_decoder as LQ
from model.multi_llm_v2 import MultiLlmModel, use_class
import torch.nn as nn
import re
import os
from utils import functions
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
from opts import parse_opts

from accelerate import init_empty_weights, load_checkpoint_and_dispatch

args = parse_opts()


os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

def to_device(data, device):
    """
    Recursively moves the input data (which can be a dictionary, list, or tensor) to the specified device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    else:
        # For other types of data (e.g., int, float), we return it unchanged
        return data
        

if args.dataset == 'CMDC':
    # 提取评估结果中的'评估结果'部分
    def extract_assessment_result(result):
        # 匹配 "评估结果:数字" 的格式
        match = re.search(r"评估结果[:：](\d)", result)
        if match:
             return round(float(match.group(1)))  # 四舍五入返回整数
    
        # 如果没有匹配到，尝试其他可能格式
        match = re.search(r"评估结果为[:：]?(\d)", result)
        if match:
            return round(float(match.group(1)))  # 四舍五入返回整数
    
        # 如果没有匹配到，尝试其他可能格式
        match = re.search(r"评估结果是[:：]?(\d)", result)  # 处理"估结果"的特殊情况
        if match:
             return round(float(match.group(1)))  # 四舍五入返回整数
    
        match = re.search(r"[为:：]?(\d)", result)  # 处理"估结果"的特殊情况
        if match:
             return round(float(match.group(1)))  # 四舍五入返回整数
    
        # 如果都没有匹配到，打印结果供调试
        print(f"未匹配到评估结果：{result}")
        return None  # 返回 None 表示未找到结果
else:
    def extract_assessment_result(key, result):
        # 1. 匹配标准格式 "Evaluation Result: 数字"（支持大小写）
        match = re.search(r"(?i)Evaluation (?i)Result: (\d)", result)  # (?i)表示忽略大小写
        if match:
            return round(float(match.group(1)))  # 四舍五入返回整数
    
        # 2. 匹配简化格式 "Result: 数字"（支持大小写）
        match = re.search(r"(?i)Result: (\d+\.?\d*)", result)
        if match:
            return round(float(match.group(1)))
    
        # 3. 匹配可能的 "Score: 数字"（支持大小写）
        match = re.search(r"(?i)Score: (\d+\.?\d*)", result)
        if match:
            return round(float(match.group(1)))
    
        # 4. 匹配孤立数字（兜底方案）
        match = re.search(r"(\d+\.?\d*)", result)
        if match:
            return round(float(match.group(1)))
    
        # 5. 如果都没有匹配到，打印结果供调试
        print(f"{key}未匹配到评估结果：{result}")
        return 1.5  # 返回均值补齐



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


# ============================== Distributed Setup ==============================

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'  # 您可以选择其他端口
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    try:
        dist.destroy_process_group()
    except AssertionError:
        print("Process group already destroyed or not initialized.")

if args.dataset == "CMDC":
    def split_data(raw_data):
        train_data = []
        test_data = []
        for key, value in raw_data.items():
            if key.startswith("HC"):
                # 提取数字部分
                num_str = key[2:].split('-')[0]  # 从索引2开始到字符串结束的部分
                # 将字符串转换为整数
                num = int(num_str)
                if 0 < num < 41:
                    train_data.append((key, value))
                else:
                    test_data.append((key, value))
            else:
                # 提取数字部分
                num_str = key[3:].split('-')[0]  # 从索引2开始到字符串结束的部分
                # 将字符串转换为整数
                num = int(num_str)
                if 0 < num < 21:
                    train_data.append((key, value))
                else:
                    test_data.append((key, value))
        return train_data, test_data
else:
    def split_data(raw_data):
        train_data = []
        test_data = []
        train_split = pd.read_csv('/root/autodl-tmp/E-DAIC-WOZ/train_split.csv')
        dev_split = pd.read_csv('/root/autodl-tmp/E-DAIC-WOZ/dev_split.csv')
        test_split = pd.read_csv('/root/autodl-tmp/E-DAIC-WOZ/test_split.csv')

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


def save_results(results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def gather_dict_results(results, world_size):
    gathered_results = [None for _ in range(world_size)]  # 确保长度正确
    
    # 收集所有进程的字典
    dist.all_gather_object(gathered_results, results)

    combined_results = [item for sublist in gathered_results for item in sublist]
    return combined_results


def train_ddp(rank, world_size, model_dir, data_path, output_path, train_model='train'):
    try:
        setup_distributed(rank, world_size)

        # 设置设备
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')

        # 设置分词器
        tokenizer = setup_tokenizer(model_dir)

        # 载入 LLaMA 模型（使用全精度）
        llama_model = LlamaForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float32,  # 切换到 float32
            device_map={"": rank}, # 将模型加载到对应的 GPU
        )

        
        # 同步模型的嵌入层大小
        llama_model.resize_token_embeddings(len(tokenizer))
        
        if args.GD_llm:
            # 冻结所有参数
            for param in llama_model.parameters():
                param.requires_grad = False
        else:
            # 设置 LoRA 配置
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                task_type="CAUSAL_LM"
            )

            # 将 LoRA 层应用到模型
            llama_model = get_peft_model(llama_model, lora_config)

        # 初始化 LQ 模型
        if args.dataset == 'CMDC':
            LQ_vid_model = LQ(d_model=710, nhead=5, num_decoder_layers=4, dim_feedforward=1024, dropout=0.1, ).to(device)
            LQ_wav_model = LQ(d_model=768, nhead=8, num_decoder_layers=4, dim_feedforward=1024, dropout=0.1, ).to(device)
        else:
            LQ_vid_model = LQ(d_model=1024, nhead=8, num_decoder_layers=4, dim_feedforward=1024, dropout=0.1, ).to(device)
            LQ_wav_model = LQ(d_model=1024, nhead=8, num_decoder_layers=4, dim_feedforward=1024, dropout=0.1, ).to(device)

        # 初始化多模态模型
        MMllm = MultiLlmModel(llama_model, tokenizer, LQ_wav_model, LQ_vid_model).to(device)

        # if args.GD_llm:
        #     s_dict = MMllm.state_dict()
        #     layer_name = []
        #     #只加载
        #     s_dict, layer_name = functions.load_checkpoint(s_dict, layer_name)
        #     MMllm.load_state_dict(s_dict)

        #固定LQ，微调LLM
        if args.LQ_former and not args.GD_llm :
            s_dict = MMllm.state_dict()
            layer_name = []#需要被固定的网络
            # s_dict, layer_name = functions.load_checkpoint(s_dict, layer_name)#####原始参数
            s_dict, layer_name = functions.load_LQ_checkpoint(s_dict, layer_name)
            #加载并固定
            MMllm = functions.load_GD(MMllm, s_dict, layer_name)
        
        #训练LQ,固定LLM
        if args.LQ_former:
            pass


        for name, param in MMllm.named_parameters():
            if param.requires_grad:
                print("放开的网络层：",name)

        # 包装为 DDP 模型
        MMllm = nn.parallel.DistributedDataParallel(MMllm, device_ids=[rank], output_device=rank,find_unused_parameters=True)

    
        # 定义优化器
        if args.LQ_former or args.GD_llm: 
            optimizer_list = [
                {'params': MMllm.module.llama_model.parameters(), 'lr': 1e-5},
                {'params': MMllm.module.wav_projection.parameters(), 'lr': 5e-4},
                {'params': MMllm.module.vid_projection.parameters(), 'lr': 5e-4},
                {'params': MMllm.module.LQ_wav_model.parameters(), 'lr': 1e-4},
            ]#未定义的默认为 1e-3
        else:
            optimizer_list = [
                {'params': MMllm.module.llama_model.parameters(), 'lr': 1e-5}
            ]
            
        if use_class:
            optimizer_list.append({'params': MMllm.module.MF.parameters(), 'lr': 1e-4})


        optimizer = optim.AdamW(optimizer_list)

        # 准备数据集和数据加载器
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        train_data, test_data = split_data(raw_data)

        # 初始化数据集
        train_dataset = MultiQnADataset(train_data, tokenizer)
        test_dataset = MultiQnADataset(test_data, tokenizer, load_test=True)
        # 打印训练集和测试集的数量
        print(f"训练集数量: {len(train_dataset)}")
        print(f"测试集数量: {len(test_dataset)}")

        # 使用 DistributedSampler
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        # 数据加载器
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler,
                                  collate_fn=train_dataset.collate_fn)
        
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, sampler=test_sampler,
                                 collate_fn=test_dataset.collate_fn)

        # 定义训练参数
        num_epochs = args.nEpochs
        log_interval = 1

        # 添加梯度裁剪
        max_grad_norm = 1.0

        if train_model == "train":
            # 开始训练循环
            for epoch in range(num_epochs):
                MMllm.train()
                train_sampler.set_epoch(epoch)
                for batch_idx, batch_data in enumerate(train_loader):
                    try:
                        optimizer.zero_grad()
                        outputs, category_loss = MMllm(to_device(batch_data, device))
                        loss = outputs.loss
                        if use_class:
                            #动态调整损失权重（前期偏向 loss，后期偏向 category_loss）
                            alpha = epoch / num_epochs  # 权重因子，从 0 增加到 1
                            total_loss = alpha * loss + (1 - alpha) * category_loss   #先训练分类，再训练文本生成
                            # total_loss = 0.9 * loss + 0.1 * category_loss
                            print(loss, category_loss)
                        else:
                            total_loss = loss
                            
                        if torch.isnan(total_loss):
                            raise ValueError("NaN loss encountered")
        
                        total_loss.backward()
                        # if batch_idx % 20 == 0:
                        #     print('************************')
                        #     #释放未使用的缓存显存
                        #     torch.cuda.empty_cache()  #导致速度很慢
        
                        # 添加梯度裁剪
                        torch.nn.utils.clip_grad_norm_(MMllm.parameters(), max_grad_norm)
        
                        optimizer.step()
        
                        if rank == 0 and batch_idx % log_interval == 0:
                            print(f"Epoch {epoch} Batch {batch_idx} Loss {total_loss.item()}")
    
                    except Exception as e:
                        if rank == 0:
                            print(f"Error occurred at Epoch {epoch} Batch {batch_idx}")
                            print(f"Exception: {e}")
                            print("Batch data keys:", batch_data.keys())
                            print("Input IDs shape:", batch_data['input_ids'].shape)
                        cleanup_distributed()
                        raise e
                        
            if args.GD_llm:
                if args.dataset == 'CMDC':
                    functions.save_model_layers(MMllm, "./checkpoint/best_CMDC_LQ_former.pth")
                else:
                    functions.save_model_layers(MMllm, "./checkpoint/best_DAIC_LQ_former.pth")
    
            # 确保所有进程完成训练
            dist.barrier()

        test_data_dict = {d[0]: d[1] for d in test_data}  # 创建一个字典，d[0]是key，d[1]是sample
        print('开始推理')
        # 模型训练完毕后开始推理
        MMllm.eval()
        with torch.no_grad():
            results = []
            print('test_loader',len(test_loader))
            for test_idx, test_sample in enumerate(test_loader):
                # 使用模型生成输出
                # try:
                generated_ids, category_label = MMllm.module.generate(to_device(test_sample, device), max_new_tokens=200)
                # except Exception as e:
                #     print(f"Error during generation for sample {test_sample['keys']}: {e}")
                #     continue

                # 将生成的 token ids 转换为文本
                generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
                print("\n模型生成的回复：")
                print(generated_texts)
                print(test_sample['keys'])

                for i in range(len(generated_texts)):
                    generated_text = generated_texts[i]
                    if use_class:
                        if extract_assessment_result(generated_texts[i]) is None:##没有提取到结果
                            if args.dataset == 'CMDC':
                                generated_text = f"评估结果：{category_label[i]}。" + generated_text
                            else:
                                generated_text = f"Evaluation Result: {category_label[i]}. " + generated_text
                                
                    key = test_sample['keys'][i]
                    result = {
                        'key': key,
                        'result': generated_text,
                        'label': test_data_dict[key]['label']  # 使用字典查找标签
                    }

                    # 使用锁来确保在访问 results 时没有其他进程同时修改
                    results.append(result)

            all_results = gather_dict_results(results, world_size)

            if rank == 0:
                save_results(all_results, output_path)
        
            print("\n测试集推理完毕，结果已保存至", output_path)

            # 确保所有进程完成推理
            dist.barrier()
    except Exception as e:
        cleanup_distributed()
        raise e


# ============================== Main Function ==============================

def main():
    world_size = 2  # 使用 GPU 2 和 GPU 3
    if args.GD_llm:
        if args.dataset == 'CMDC':
            data_path = '/root/autodl-tmp/CMDC/output_data03.json'  # 训练数据路径
        else:
            data_path = '/root/autodl-tmp/E-DAIC-WOZ/output03.json'  # 训练数据路径
    else:
        if args.dataset == 'CMDC':
            data_path = '/root/autodl-tmp/CMDC/output_data02.json'  # 训练数据路径
        else:
            data_path = '/root/autodl-tmp/E-DAIC-WOZ/output.json'  # 训练数据路径
            
    output_path = f'./{args.dataset}_{args.LQ_former}_{args.GD_llm}_{args.use_class}_test_results_with_generated_output.json'  # 推理结果保存路径

    if args.dataset == 'CMDC':
        model_dir = '/root/MLlm-DR/llama-3-chinese-8b-instruct'  # 请替换为您的模型路径
    else:
        model_dir = '/root/autodl-tmp/llama-3-8b-instruct'
    
    # 指定可见的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    train_model = args.train_model
    mp.spawn(train_ddp,
             args=(world_size, model_dir, data_path, output_path, train_model),
             nprocs=world_size,
             join=True)
    

if __name__ == "__main__":
    main()
