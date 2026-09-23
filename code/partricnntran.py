import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import DropPath

## 拼接在Vits前的卷积模块
class Conv_Layers(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * 4, (1, 3), stride=1, padding=(0, 1))
        self.conv2 = nn.Conv2d(in_channels * 4, in_channels, (1, 3), stride=1, padding=(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        return out

        
## 将输入数据分割为patch，并将每个patch转换为嵌入向量，同时添加位置编码
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size_x: int = 1, patch_size_y: int = 6, emb_size: int = 64, img_size: int = 120, num_cls_tokens: int = 6):
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y
        self.num_cls_tokens = num_cls_tokens  # 保存需要的CLS token数量
        self.positions_num = (img_size // patch_size_y) + num_cls_tokens  # 根据CLS token的数量调整位置编码的大小
        
        super().__init__()
        
        # 定义卷积层和位置编码
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=(patch_size_x, patch_size_y), stride=(patch_size_x, patch_size_y)),
            nn.BatchNorm2d(emb_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )       

        # 增加num_cls_tokens个cls_token
        self.cls_token_a = nn.Parameter(torch.randn(1, num_cls_tokens//2, emb_size)) 
        self.cls_token_b = nn.Parameter(torch.randn(1, num_cls_tokens//2, emb_size)) 
        
        # 位置编码，考虑CLS token的位置
        self.positions = nn.Parameter(torch.randn(self.positions_num, emb_size))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)  # 将原始图像转换为patch嵌入
        
        cls_tokens_a = repeat(self.cls_token_a, '() n e -> b n e', b=b)  #生成cls_token，并将其维度扩展至batch_size
        cls_tokens_b = repeat(self.cls_token_b, '() n e -> b n e', b=b)  #生成cls_token，并将其维度扩展至batch_size
        x = torch.cat([cls_tokens_a, x], dim=1)  #把cls_token和patch在第2个维度（patch）前进行拼接
        x = torch.cat([x, cls_tokens_b], dim=1)  #把cls_token和patch在第2个维度（patch）后进行拼接
        
        x += self.positions  # 添加位置编码
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 64, num_heads: int = 8):
        super().__init__()
        self.emb_size = emb_size  #保存嵌入维度
        self.num_heads = num_heads  #保存多头注意力的头数
        self.qkv = nn.Linear(emb_size, emb_size * 3)  #定义一个线性层，用于把输入映射为qkv
        self.projection = nn.Linear(emb_size, emb_size)  #定义一个线性层，用于把多头注意力的输出映射为原始维度
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        ## 得到qkv
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h = self.num_heads, qkv = 3)  #计算qkv之后重组维度
        queries, keys, values = qkv[0], qkv[1], qkv[2]  #分别提取qkv

        ## 计算注意力分数
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  #定义乘法方式，计算QK之间的点积
        scaling = self.emb_size ** (1/2)  #计算缩放因子，对注意力得分进行缩放，通常是嵌入维度的平方根
        att = F.softmax(energy, dim=-1) / scaling  #对注意力得分进行softmax，将得分转换为概率分布
        
        ## 计算注意力输出
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)  #使用注意力权重处理values
        out = rearrange(out, "b h n d -> b n (h d)")  #将多头的输出拼接在一起
        out = self.projection(out)  #将多头注意力的输出映射回原始维度
        return out
    
## 实现了一个残差连接
class ResidualAdd(nn.Module):
    ## fn是一个子网络模块或者一系列操作，总之是一个有输入输出的东西
    ## 将传入的fn保存为类的属性，从而在forward方法中使用
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    ## 定义残差块前向传播方法
    def forward(self, x, **kwargs):
        res = x  #保存输入为残差
        x = self.fn(x, **kwargs)  #调用fn，计算输出
        x += res  #输出+残差，得到残差块结果
        return x

## transformer编码器中的一个组件，前馈神经网络
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4):  #expansion: 前馈网络的扩展倍数
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),  #线性层，将输入映射到更高维度
            nn.GELU(),  #激活函数，gelu而不是relu
            nn.Linear(expansion * emb_size, emb_size),  #线性层，将高维度映射回原始维度
        )

## transformer编码器的基本组成部分，堆叠形成transformer编码器
## 包含一个多头注意力模块和一个前馈模块
class TransformerEncoderBlock(nn.Sequential):
    ## forward_expansion: 前馈模块的扩展倍数，指示了前馈网络中间层比输入层大几倍
    def __init__(self,
                 emb_size: int = 64,
                 forward_expansion: int = 4,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),  #层归一化
                MultiHeadAttention(emb_size, **kwargs),  #多头注意力模块
                DropPath(0.2)  # 添加 DropPath
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),  #层归一化
                FeedForwardBlock(emb_size, expansion=forward_expansion),    #前馈模块
                DropPath(0.2)  # 添加 DropPath
            )
            ))

## 继承自nn.Sequential，将多个transformerEncoderBlock堆叠到一起，形成transformerEncoder
class TransformerEncoder(nn.Sequential):
    ## depth定义了堆叠几个transformerEncoderBlock
    def __init__(self, depth: int = 4, emb_size = 64, **kwargs):
        ## __init__调用父类构造函数，将多个transformerEncoderBlock组合到一起
        ## 使用了列表生成式创建一个包含depth个TransformerEncoderBlock的列表。解包后传给父类的构造函数
        super().__init__(
            nn.Dropout(0.25),
            *[TransformerEncoderBlock(emb_size, **kwargs) for _ in range(depth)],
            nn.LayerNorm(emb_size)
        )

## 模型的最后一层，将transformer的输出映射到类别
class ClassificationHead(nn.Module):
    def __init__(self, emb_size=64, num_classes_per_token=None, dropout_rate=0.25, use_sigmoid=True):
        super(ClassificationHead, self).__init__()
        if num_classes_per_token is None:
            num_classes_per_token = [15, 15, 15, 15, 15, 15, 15, 16]
        self.use_sigmoid = use_sigmoid
        self.num_cls_tokens = len(num_classes_per_token)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_size, emb_size * 4),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(emb_size * 4, emb_size * 2),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(emb_size * 2, num_classes)
            ) for num_classes in num_classes_per_token
        ])

    def forward(self, x):
        cls_token_first = x[:, :self.num_cls_tokens//2, :]
        cls_token_last = x[:, -self.num_cls_tokens//2:, :]
        cls_tokens = torch.cat((cls_token_first, cls_token_last), dim=1)
        outputs = []
        for i, head in enumerate(self.heads):
            outputs.append(head(cls_tokens[:, i, :]))
        combined_outputs = torch.cat([output for output in outputs], dim=-1)
        if self.use_sigmoid:
            return torch.sigmoid(combined_outputs)
        return combined_outputs

## label -> (输出格数列表, 是否加 sigmoid)
CASE = {
    'StairCase': ([60, 60, 60, 60, 60, 60, 60, 60], True),
    'Gauss':     ([60, 60, 60, 60, 60, 60, 60, 60], False),
    'Hard':      ([15, 15, 15, 15, 15, 15, 15, 16], False),
    'BCE':       ([15, 15, 15, 15, 15, 15, 15, 16], False),
    'Stair121':  ([15, 15, 15, 15, 15, 15, 15, 16], True),
    'Stair1200': ([150, 150, 150, 150, 150, 150, 150, 150], True),
    'Gauss1200': ([150, 150, 150, 150, 150, 150, 150, 150], False),
}

class partricnntran(nn.Sequential):
    def __init__(self,
                in_channels: int = 3,
                patch_size_x: int = 1,
                patch_size_y: int = 6,
                emb_size: int = 128,
                img_size: int = 120,
                depth: int = 4,
                n_classes: int = 121,
                num_cls_tokens: int = 8,
                case1: str = 'Gauss',
                **kwargs):
        list_fc, use_sig = CASE[case1]
        super().__init__(
            Conv_Layers(in_channels),
            PatchEmbedding(in_channels, patch_size_x, patch_size_y, emb_size, img_size, num_cls_tokens=num_cls_tokens),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, num_classes_per_token=list_fc, use_sigmoid=use_sig)
        )
