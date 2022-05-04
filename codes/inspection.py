import numpy as np
import torch
from torch.utils.data import DataLoader

from codes import mvtecad
from codes.logger import Logger
from .utils import NHWC2NCHW, distribute_scores, PatchDataset_NCHW

__all__ = ['eval_encoder_NN_multiK', 'eval_embeddings_NN_multiK']

logger = Logger().getLogger()


def infer(x, enc, K, S):
    x = NHWC2NCHW(x)
    # 处理成把图整齐切块的dataset（unravel_index那个函数）
    dataset = PatchDataset_NCHW(x, K=K, S=S)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True)
    embs = np.empty((dataset.N, dataset.row_num, dataset.col_num, enc.D), dtype=np.float32)  # [-1, I, J, D]
    enc = enc.eval()
    with torch.no_grad():
        for xs, ns, iis, js in loader:
            xs = xs.cuda(1)
            # 对图中每个位置都切块，对每个块进行encode
            embedding = enc(xs)
            # b, d = embedding.size()
            # embedding = embedding.reshape(b, -1)
            embedding = embedding.detach().cpu().numpy()
            # 拼成原图大小
            for embed, n, i, j in zip(embedding, ns, iis, js):
                embs[n, i, j] = np.squeeze(embed)
    return embs


def assess_anomaly_maps(obj, anomaly_maps):
    auroc_seg = mvtecad.segmentation_auroc(obj, anomaly_maps)

    anomaly_scores = anomaly_maps.max(axis=-1).max(axis=-1)
    auroc_det = mvtecad.detection_auroc(obj, anomaly_scores)
    return auroc_det, auroc_seg


#########################

def eval_encoder_NN_multiK(enc_64, enc_32, obj):
    logger.info("----------evaluating------------")
    # load 数据
    x_tr = mvtecad.get_x_standardized(obj, mode='train')
    x_te = mvtecad.get_x_standardized(obj, mode='test')
    # 生成不同尺度的embedding
    # [B,13,13,32]  (batch,conv_weight,conv_height,patch_size)
    embs16_tr = infer(x_tr, enc_64, K=64, S=16)
    embs16_te = infer(x_te, enc_64, K=64, S=16)

    x_tr = mvtecad.get_x_standardized(obj, mode='train')
    x_te = mvtecad.get_x_standardized(obj, mode='test')
    # [B,57,57,32]
    embs32_tr = infer(x_tr, enc_32, K=32, S=8)
    embs32_te = infer(x_te, enc_32, K=32, S=8)

    embs16 = embs16_tr, embs16_te
    embs32 = embs32_tr, embs32_te

    return eval_embeddings_NN_multiK(obj, embs16, embs32)


def eval_embeddings_NN_multiK(obj, embs16, embs32, NN=1):
    emb_tr, emb_te = embs16
    # [B,13,13]
    # 距离最近的正常的patch的距离（用的KDtree求最邻近，这个你可能要查一下）
    maps_16 = measure_emb_NN(emb_te, emb_tr, method='kdt', NN=NN)
    # [B,256,256]
    # 按照patch_svdd论文中第八页说的算公式
    maps_16 = distribute_scores(maps_16, (256, 256), K=64, S=16)  # 分配到每个像素上
    # 计算检测以及分割的auc
    det_16, seg_16 = assess_anomaly_maps(obj, maps_16)

    # 同理 不同尺度
    emb_tr, emb_te = embs32
    maps_32 = measure_emb_NN(emb_te, emb_tr, method='ngt', NN=NN)
    maps_32 = distribute_scores(maps_32, (256, 256), K=32, S=8)
    det_32, seg_32 = assess_anomaly_maps(obj, maps_32)

    maps_sum = maps_16 + maps_32
    det_sum, seg_sum = assess_anomaly_maps(obj, maps_sum)

    maps_mult = maps_16 * maps_32  # equation 9
    det_mult, seg_mult = assess_anomaly_maps(obj, maps_mult)

    return {
        'det_16': det_16,
        'seg_16': seg_16,

        'det_32': det_32,
        'seg_32': seg_32,

        'det_sum': det_sum,
        'seg_sum': seg_sum,

        'det_mult': det_mult,
        'seg_mult': seg_mult,

        'maps_16': maps_16,
        'maps_32': maps_32,

        'maps_sum': maps_sum,
        'maps_mult': maps_mult,
    }


########################

def measure_emb_NN(emb_te, emb_tr, method='kdt', NN=1):
    from .nearest_neighbor import search_NN
    D = emb_tr.shape[-1]
    train_emb_all = emb_tr.reshape(-1, D)

    # 找离正常patch最近的距离
    l2_maps, _ = search_NN(emb_te, train_emb_all, method=method, NN=NN)
    anomaly_maps = np.mean(l2_maps, axis=-1)
    # 「 b,w,d,c 」
    return anomaly_maps
