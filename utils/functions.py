import torch
import os
from collections import OrderedDict
from opts import parse_opts
args = parse_opts()

def load_checkpoint(s_dict, layer_name):
    if args.dataset == 'CMDC':
        checkpoint = torch.load("./checkpoint/best_model_TF_visual_59.pth")
    else:
        checkpoint = torch.load("./checkpoint/best_model_DAIC_109_visual.pth")
    
    s_dict['vid_tgt'] = checkpoint['tgt']#固定tgt参数
    layer_name.append('vid_tgt')
    for name in checkpoint:
        name_ = "LQ_vid_model" + name.replace("former", "")
        if name_ in s_dict:
            s_dict[name_] = checkpoint[name]
            layer_name.append(name_)
    print('加载视觉特征提取网络')

    if args.dataset == 'CMDC':
        checkpoint = torch.load("./checkpoint/best_model_TF_wav_69.pth")
    else:
        checkpoint = torch.load("./checkpoint/best_model_DAIC_108_wav.pth")
    s_dict['wav_tgt'] = checkpoint['tgt']
    layer_name.append('wav_tgt')
    for name in checkpoint:
        name_ = "LQ_wav_model" + name.replace("former", "")
        if name_ in s_dict:
            s_dict[name_] = checkpoint[name]
            layer_name.append(name_)
    print('加载语音特征提取网络')
    
    return s_dict,layer_name



def load_LQ_checkpoint(s_dict, layer_name):
    # 加载保存的模型参数
    checkpoint = torch.load(f"./checkpoint/best_{args.dataset}_LQ_former.pth")

    # 遍历 checkpoint 中的每个层参数
    for name, params in checkpoint.items():
        if isinstance(params, OrderedDict):# 如果是 OrderedDict（即子模块的参数），递归加载
            for sub_layer_name, sub_params in params.items():
                name_ = name +'.'+ sub_layer_name#定位层名
                s_dict[name_] = sub_params
                layer_name.append(name_)
        else:
            # 如果是普通参数，直接加载
            s_dict[name] = params
            layer_name.append(name)

    # 返回更新后的模型参数字典
    return s_dict,layer_name



# 加载并固定网络参数
def load_GD(model,s_dict,layer_name):
    # print(layer_name)
    # 加载
    model.load_state_dict(s_dict)
    for (name, param) in model.named_parameters():
        if name in layer_name:  #
            # print(name, "被固定的网络层")
            param.requires_grad = False  # 固定
        else:
            # print(name, "被放开的网络层")
            pass
    return model


def save_model_layers(model, output_path):
    # 保存指定层的网络参数
    layers_to_save = {
        "LQ_wav_model": model.module.LQ_wav_model.state_dict(),
        "LQ_vid_model": model.module.LQ_vid_model.state_dict(),
        "wav_projection": model.module.wav_projection.state_dict(),
        "vid_projection": model.module.vid_projection.state_dict(),
        "wav_tgt": model.module.wav_tgt,
        "vid_tgt": model.module.vid_tgt,
    }

    # 保存所有层的参数到一个字典
    torch.save(layers_to_save, output_path)
    print(f"Saved model layers to {output_path}")


