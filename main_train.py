import argparse
import time

import torch
from torch import nn
from codes import mvtecad
from functools import reduce
from torch.utils.data import DataLoader
import torch.distributed as dist
from codes.datasets import *
from codes.networks import *
from codes.inspection import eval_encoder_NN_multiK
from codes.utils import *
from codes.vit import Transformer, ViT
# from linformer import Linformer
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--obj', default='wood', type=str)
parser.add_argument('--lambda_value', default=1, type=float)
parser.add_argument('--D', default=32, type=int)

parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
# # parser.add_argument('--weights', type=str, default='./vit_base_patch16_224_in21k.pth',
# #                         help='initial weights path')
args = parser.parse_args()
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# efficient_transformer = Linformer(
#     dim=32,
#     seq_len=10,  # 3*3 patches + 1 cls-token
#     depth=12,
#     heads=8,
#     k=32
# )


def train():
    obj = args.obj
    D = args.D
    lr = args.lr
    # print(torch.cuda.device_count())

    # print(device)
    # 搭建网络结构

    with task('Networks'):
        # enc = EncoderHier(32, D).cuda(1)  # 八层卷积
        # cls_32 = PositionClassifier(32, D).cuda(1)  # 全连接分类
        # cls_32 = PositionClassifier(32, D).cuda(1)  # 全连接分类
        ViT_64 = ViT(
            image_size=192,
            patch_size=16,
            channels=8,
            dim=64,
            depth=24,
            heads=16,
            mlp_dim=128,
            dropout=0.1,
            emb_dropout=0.1
        ).to(device)

        ViT_32 = ViT(
            image_size=96,
            patch_size=8,
            channels=8,
            dim=64,
            depth=24,
            heads=16,
            mlp_dim=128,
            dropout=0.1,
            emb_dropout=0.1
        ).to(device)

        modules = [ViT_64, ViT_32]
        params = [list(module.parameters()) for module in modules]
        params = reduce(lambda x, y: x + y, params)

        opt = torch.optim.AdamW(params=params, lr=lr)

    with task('Datasets'):
        train_x = mvtecad.get_x_standardized(obj, mode='train')  # 取出数据集中的图片并进行标准化
        train_x = NHWC2NCHW(train_x)  # 把[batch，长，宽，channel]转换成[batch，channel，长，宽]

        rep = 100
        datasets = dict()
        datasets['pos_64'] = PositionDataset(train_x, K=64, repeat=rep)
        datasets['pos_32'] = PositionDataset(train_x, K=32, repeat=rep)

        dataset = DictionaryConcatDataset(datasets)
        train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

    print('Start training')
    for i_epoch in range(args.epochs):
        # if i_epoch != 0:
        start = time.time()

        print('epoch %d:' % i_epoch)
        for module in modules:
            module.train()

        for loader in train_loader:
            loader = to_device(loader, device)
            loss_pos_64 = ViT_64(loader['pos_64'])
            # loss_pos_8 = ViT_8(loader['pos_8'])
            loss_pos_32 = ViT_32(loader['pos_32'])

            loss = loss_pos_64 + loss_pos_32
            print("loss:%f" % loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
        aurocs = eval_encoder_NN_multiK(enc_64=ViT_64, enc_32=ViT_32, obj=obj)
        end = time.time()
        cost_time = end - start
        log_result(obj, aurocs, cost_time)
        # ViT_32.save(obj, 32)
        ViT_64.save(obj, 64)
        ViT_32.save(obj, 32)
        # ViT_8.save(obj, 8)
    print("Training end")


def log_result(obj, aurocs, cost_time):
    det_16 = aurocs['det_16'] * 100
    seg_16 = aurocs['seg_16'] * 100
    det_32 = aurocs['det_32'] * 100
    seg_32 = aurocs['seg_32'] * 100
    #
    #
    det_sum = aurocs['det_sum'] * 100
    seg_sum = aurocs['seg_sum'] * 100

    det_mult = aurocs['det_mult'] * 100
    seg_mult = aurocs['seg_mult'] * 100

    print(
        f'|K16| Det: {det_16:4.1f} Seg: {seg_16:4.1f} '
        f'|K32| Det: {det_32:4.1f} Seg: {seg_32:4.1f} '
        f'|sum| Det: {det_sum:4.1f} Seg: {seg_sum:4.1f} '
        f'|mult| Det: {det_mult:4.1f} Seg: {seg_mult:4.1f} '
        f'({obj}) '
        f'cost_time:{cost_time}')


if __name__ == '__main__':
    train()
