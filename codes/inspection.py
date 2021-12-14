from codes import mvtecad
import numpy as np
import torch
from torch.utils.data import DataLoader
from .utils import NHWC2NCHW, distribute_scores
from codes.datasets import VisionDataset

__all__ = ['eval_encoder_NN_multiK', 'eval_embeddings_NN_multiK']


def infer(x, enc, K):
    x = NHWC2NCHW(x)
    dataset = VisionDataset(x, K=K)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True)
    embs = np.empty((dataset.N, dataset.row_num, dataset.col_num, enc.D), dtype=np.float32)  # [-1, I, J, D]
    count = 0
    enc = enc.eval()
    with torch.no_grad():
        for xs in loader:
            xs = xs.cuda(1)
            embedding = enc(xs)
            b, n, d = embedding.size()
            embedding = embedding.reshape(b, dataset.row_num, dataset.col_num, enc.D)
            embedding = embedding.detach().cpu().numpy()
            embs[count:count + b, :, :, :] = embedding
            count = count + b
    return embs


def assess_anomaly_maps(obj, anomaly_maps):
    auroc_seg = mvtecad.segmentation_auroc(obj, anomaly_maps)

    anomaly_scores = anomaly_maps.max(axis=-1).max(axis=-1)
    auroc_det = mvtecad.detection_auroc(obj, anomaly_scores)
    return auroc_det, auroc_seg


#########################

def eval_encoder_NN_multiK(enc_16, obj):
    print("----------evaluating------------")
    x_tr = mvtecad.get_x_standardized(obj, mode='train')
    x_te = mvtecad.get_x_standardized(obj, mode='test')

    # [B,13,13,64]  (batch,conv_weight,conv_height,patch_size)
    embs16_tr = infer(x_tr, enc_16, K=16)
    embs16_te = infer(x_te, enc_16, K=16)

    # x_tr = mvtecad.get_x_standardized(obj, mode='train')
    # x_te = mvtecad.get_x_standardized(obj, mode='test')
    # # [B,57,57,64]
    # embs8_tr = infer(x_tr, enc_8, K=8)
    # embs8_te = infer(x_te, enc_8, K=8)

    embs16 = embs16_tr, embs16_te
    # embs8 = embs8_tr, embs8_te

    return eval_embeddings_NN_multiK(obj, embs16)


def eval_embeddings_NN_multiK(obj, embs16, NN=1):
    emb_tr, emb_te = embs16
    # [B,13,13]
    maps_16 = measure_emb_NN(emb_te, emb_tr, method='kdt', NN=NN)
    # 距离最近的正常的patch的距离
    # [B,256,256]
    maps_16 = distribute_scores(maps_16, (256, 256), K=64, S=16)  # 分配到每个像素上
    det_16, seg_16 = assess_anomaly_maps(obj, maps_16)

    # emb_tr, emb_te = embs8
    # maps_8 = measure_emb_NN(emb_te, emb_tr, method='ngt', NN=NN)
    # maps_8 = distribute_scores(maps_8, (256, 256), K=32, S=4)
    # det_8, seg_8 = assess_anomaly_maps(obj, maps_8)
    #
    # maps_sum = maps_16 + maps_8
    # det_sum, seg_sum = assess_anomaly_maps(obj, maps_sum)
    #
    # maps_mult = maps_16 * maps_8  # equation 9
    # det_mult, seg_mult = assess_anomaly_maps(obj, maps_mult)

    return {
        'det_16': det_16,
        'seg_16': seg_16,

        # 'det_8': det_8,
        # 'seg_8': seg_8,
        #
        # 'det_sum': det_sum,
        # 'seg_sum': seg_sum,
        #
        # 'det_mult': det_mult,
        # 'seg_mult': seg_mult,
        #
        # 'maps_16': maps_16,
        # 'maps_8': maps_8,
        #
        # 'maps_sum': maps_sum,
        # 'maps_mult': maps_mult,
    }


########################

def measure_emb_NN(emb_te, emb_tr, method='kdt', NN=1):
    from .nearest_neighbor import search_NN
    D = emb_tr.shape[-1]
    train_emb_all = emb_tr.reshape(-1, D)

    # 找离正常patch最近的距离
    l2_maps, _ = search_NN(emb_te, train_emb_all, method=method, NN=NN)
    anomaly_maps = np.mean(l2_maps, axis=-1)
    #「 b,w,d,c 」
    return anomaly_maps
