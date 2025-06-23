import copy
from typing import Optional
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor

class Transformer_decoder(nn.Module):

    def __init__(self, d_model=512, nhead=8,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,return_intermediate=return_intermediate_dec)

        self._reset_parameters()
        # 位置编码
        self.pos_emb = PositionalEncoding_(d_model)#
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, tgt, memory, memory_mask):
        # 位置编码
        query_embed = self.pos_emb(tgt)##位置编码器
        pos_embed = self.pos_emb(memory)##位置编码器

        x = self.decoder(tgt, memory, memory_key_padding_mask=memory_mask, pos=pos_embed, query_pos=query_embed)#没有传递 tgt_mask 参数，模型会默认使用 tgt_mask=None。这意味着在解码器的 自注意力（Self-Attention） 模块中，目标序列的所有位置可以自由地相互关注，不会进行倒三角遮掩（causal masking）。

        return x


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate


    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None
                ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        # return output.unsqueeze(0)
        return output

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)#固定维度
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)#512
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        # tgt 自注意力编码
        q = k = self.with_pos_embed(tgt, query_pos)#位置编码
        #被传入 None，默认情况下不会应用任何遮掩操作，这等效于掩码矩阵全为零。
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)#
        # print('memory_key_padding_mask', memory_key_padding_mask.size())#(32,10)
        # tgt与memory进行跨模态注意力编码
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def generate_causal_mask(seq_len):
    # 生成倒三角掩码
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)  # 上三角为1
    return mask.masked_fill(mask == 1, float('-inf'))  # 用 -inf 进行填充


############位置编码
class PositionalEncoding_(nn.Module):
    def __init__(self, d_embed, seq_len=8000):
        super(PositionalEncoding_, self).__init__()
        self.d_model = d_embed
        pe = torch.zeros(seq_len, d_embed)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_embed, 2).float()
            * (-math.log(10000.0) / d_embed)
        )
        pe[:, 0::2] = torch.sin(position * div_term)# 字嵌入维度为偶数时
        pe[:, 1::2] = torch.cos(position * div_term)# 字嵌入维度为奇数时
        pe = pe.unsqueeze(0)## 在指定维度0上增加维度大小为1[3,4] -> [1,3,4]
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)# sqrt() 方法返回数字x的平方根 适应多头注意力机制的计算，它说“这个数值的选取略微增加了内积值的标准差，从而降低了内积过大的风险，可以使词嵌入张量在使用多头注意力时更加稳定可靠”
        # print("x",x.size())
        x_pos = self.pe[:, : x.size(1), :]# 变为x.size(1)长度，torch.Size([1, 4, 512])
        # print("x_pos",x_pos.size())
        # x = x + x_pos#torch.Size([1, 4, 512])+torch.Size([4, 4, 512]) 每一个特征都加上位置信息
        # print("x",x.size())
        return x_pos#layer层会再加上位置信息


if __name__ == '__main__':

    model = Transformer_decoder(d_model=256, nhead=8,num_decoder_layers=1, dim_feedforward=1024, dropout=0.1,)
    # 输入数据
    batch_size = 32
    seq_len = 3
    d_model = 256
    #在训练期间通过教师强制（Teacher Forcing）的方式提供给解码器。这意味着目标输入只包含预测之前的信息，用于帮助解码器预测下一个值。
    tgt = torch.rand(batch_size, seq_len, d_model)  # 目标序列的嵌入表示
    memory_key_padding_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
    src = torch.rand(batch_size, seq_len, d_model)

    # 训练过程
    output = model(tgt, src, memory_key_padding_mask)

    print(output.size())