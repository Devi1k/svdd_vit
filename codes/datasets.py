import numpy as np
from torch.utils.data import Dataset
from .utils import *

__all__ = ['SVDD_Dataset', 'PositionDataset']


def generate_coords(H, W, K):
    h = np.random.randint(0, H - K + 1)
    w = np.random.randint(0, W - K + 1)
    return h, w


def generate_coords_position(H, W, K):
    with task('P1'):
        p1 = generate_coords(H, W, K)
        h1, w1 = p1

    # pos = np.arange(8)

    with task('P'):
        J = K // 4
        K3_4 = 3 * K // 4
        # for i in range(8):
        h_dir, w_dir = pos_to_diff[0]
        h_del, w_del = np.random.randint(J, size=2)

        h_diff = h_dir * (h_del + K3_4)
        w_diff = w_dir * (w_del + K3_4)

        h2 = h1 + h_diff
        w2 = w1 + w_diff

        h2 = np.clip(h2, 0, H)
        w2 = np.clip(w2, 0, W)
        p2 = (h2, w2)

        # logger.info(p2)
        # tuple(p2)
        # logger.info(p2)
    return p2


def generate_coords_svdd(H, W, K):
    with task('P1'):
        p1 = generate_coords(H, W, K)
        h1, w1 = p1
    with task('P2'):
        J = K // 32

        p2 = []

        for i in range(8):
            h_jit, w_jit = pos_to_diff[i]
            h_jit, w_jit = J * h_jit, J * w_jit
            h2 = h1 + h_jit
            w2 = w1 + w_jit

            # 使坐标保持在图像范围以内
            h2 = np.clip(h2, 0, H - K)
            w2 = np.clip(w2, 0, W - K)
            p2.append((h2, w2))

    return p1, p2


pos_to_diff = {
    0: (-1, -1),
    1: (-1, 0),
    2: (-1, 1),
    3: (0, -1),
    4: (0, 1),
    5: (1, -1),
    6: (1, 0),
    7: (1, 1)
}


class SVDD_Dataset(Dataset):
    def __init__(self, memmap, K=64, repeat=1):
        super().__init__()
        self.arr = np.asarray(memmap)
        self.K = K
        self.repeat = repeat

    def __len__(self):
        N = self.arr.shape[0]
        return N * self.repeat

    def __getitem__(self, idx):
        N = self.arr.shape[0]
        K = self.K
        n = idx % N

        p1, p2 = generate_coords_svdd(256, 256, K)

        image = self.arr[n]

        patch1 = crop_image_CHW(image, p1, K)
        patch2 = []
        for j in p2:
            patch2.append(crop_image_CHW(image, j, K))
        image = patch2[0]
        for i in range(1, 3):
            image = np.concatenate((image, patch2[i]), axis=2)

        image1 = patch2[3]
        image1 = np.concatenate((image1, patch1), axis=2)
        image1 = np.concatenate((image1, patch2[4]), axis=2)

        image2 = patch2[5]
        for i in range(6, 8):
            image2 = np.concatenate((image2, patch2[i]), axis=2)

        image_cut = np.concatenate((image, image1, image2), axis=1)
        return image_cut


class PositionDataset(Dataset):
    def __init__(self, x, K=64, repeat=1):
        super(PositionDataset, self).__init__()
        self.x = np.asarray(x)
        self.K = K
        self.repeat = repeat

    def __len__(self):
        N = self.x.shape[0]
        return N * self.repeat

    def __getitem__(self, idx):
        N = self.x.shape[0]
        K = self.K
        n = idx % N

        image = self.x[n]
        p = generate_coords_position(256, 256, K)  # 求出p1以及其余七个格坐标

        patch_image = crop_image_CHW(image, p, K)

        rgbshift = np.random.normal(scale=0.02, size=(3, 1, 1))
        patch_image += rgbshift

        noise1 = np.random.normal(scale=0.02, size=(3, K * 3, K * 3))
        patch_image += noise1

        return patch_image
