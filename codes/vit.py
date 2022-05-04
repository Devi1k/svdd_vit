import os

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum

from codes.utils import *

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# helpers
def mask(mask_list, b, n):
    mask = np.ones((b, n, 1), dtype='float32')
    for i in range(b):
        mask[i][mask_list[i]] = 0
    mask = torch.from_numpy(mask).to(torch.float32).to('cuda:1')
    return mask


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((2 * self.window_size - 1) * (2 * self.window_size - 1), heads))  # 2*Wh-1 * 2*Ww-1, nH
        #
        # # get pair-wise relative position index for each token inside the window
        # coords_h = torch.arange(self.window_size)
        # coords_w = torch.arange(self.window_size)
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        # relative_coords[:, :, 1] += self.window_size - 1
        # relative_coords[:, :, 0] *= 2 * self.window_size - 1
        # relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # self.register_buffer("relative_position_index", relative_position_index)
        # head个数
        self.heads = heads
        # 公式里的除根号d，为了是防止输入softmax的内积过大，导致偏导数为0，选择根号d是因为能够是q*k的期望为0，方差为1类似于归一化
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        # 三个权重矩阵乘出来得到Q，K，V
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # B_, N, C, h = *x.shape, self.heads
        # qkv = self.to_qkv(x).reshape(B_, N, 3, h, C // h).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        #
        # q = q * self.scale
        # attn = (q @ k.transpose(-2, -1))

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # 拆分乘三个
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        # 公式里的Q乘K转置再除根号k
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # attn = dots + relative_position_bias.unsqueeze(0)
        # softmax一下
        attn = self.attend(dots)
        # 公式里的乘V后得到C输出
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        # 按层数堆叠attention block，每个block包括一个attention和一个全连接，每个ateention以及全连接之后都要norm+残差连接（正则化一下）
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            # 加x就是残差连接，解决梯度消失or爆炸问题
            x = attn(x) + x
            x = ff(x) + x
        return x


def makedirpath(fpath: str):
    dpath = os.path.dirname(fpath)
    if dpath:
        os.makedirs(dpath, exist_ok=True)


class ViT(nn.Module):
    def __init__(self, *, image_size, dim, depth, heads, mlp_dim, pool='mean', dim_head=64, dropout=0.,
                 emb_dropout=0.,
                 patch_size, channels):
        super().__init__()
        # 图像宽高
        image_height, image_width = pair(image_size)
        # patch一块宽高
        patch_height, patch_width = pair(patch_size)
        #
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        self.D = dim
        # patch的个数
        num_patches = (image_height // 4 // patch_height) * (image_width // 4 // patch_width)
        # window_size = image_height // patch_height // 4
        # num_patches = 9
        # patch维度：通道数*宽高
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # 先卷积一下
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, stride=(2, 2)),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=1, stride=(2, 2))
        )
        # 卷积后embedding，并且全连接降维
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )
        # 加pos_embedding,保证位置信息的学习
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        # transformer
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, channels)
        # )
        # self.rearrange = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_height, p2=patch_width)

    def forward(self, x):
        # x, _ = image
        x = self.conv(x)
        x = self.to_patch_embedding(x)
        b, n, d = x.shape

        if self.training:
            # 训练时随机选九块中的一块
            mask_list = []
            for i in range(b):
                mask_id = np.random.randint(9)
                mask_list.append(mask_id)
            # 把选中的块置零（变黑）
            mask = np.ones((b, n, 1), dtype='float32')
            for i in range(b):
                mask[i][mask_list[i]] = 0
            mask = torch.from_numpy(mask).to(torch.float32).to('cuda:1')
            # 原始状态
            x_ori = x.mul((1 - mask))
            # 点乘后得到mask的
            x = x * mask
            x += self.pos_embedding[:, :n]
            x = self.dropout(x)
            x = self.transformer(x)
            x = self.to_latent(x)
            # 将mask后预测的与原始比较做差，取l2范数
            x_cur = x.mul((1 - mask))
            diff = x_ori - x_cur
            l2 = diff.norm(dim=1)
            loss_diff = l2.mean()
            return loss_diff
        else:
            # eval时直接mask中间的那块，其余部分顺序相同
            mask_list = []
            mask_id = 4
            for i in range(b):
                mask_list.append(mask_id)
            mask = np.ones((b, n, 1), dtype='float32')
            for i in range(b):
                mask[i][mask_list[i]] = 0
            mask = torch.from_numpy(mask).to(torch.float32).to('cuda:1')
            # x_ori = x.mul((1 - mask))
            x += self.pos_embedding[:, :n]
            x = self.dropout(x)
            x = self.transformer(x)
            x = self.to_latent(x)
            x_cur = x.mul((1 - mask))
            # 只需要中间那一块
            res = x_cur[:, mask_id, :]
            return res

    def save(self, name, K):
        fpath = self.fpath_from_name(name, K)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name, K):
        fpath = self.fpath_from_name(name, K)
        self.load_state_dict(torch.load(fpath, map_location=lambda storage, loc: storage.cuda(1)))

    @staticmethod
    def fpath_from_name(name, K):
        return f'ckpts/{name + str(K)}/enchier.pkl'
