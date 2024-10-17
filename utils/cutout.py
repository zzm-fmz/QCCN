import numpy as np
import torch


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, img_size ,cut_ratio):
        self.n_holes = n_holes
        self.length = img_size * cut_ratio

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W). [50, 3, 84, 84]
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        b = img.size(0)
        h = img.size(2)
        w = img.size(3)
        for k in range(b):
            mask = np.ones((h, w), np.float32)
            for n in range(self.n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = int(np.clip(y - self.length // 2, 0, h))
                y2 = int(np.clip(y + self.length // 2, 0, h))
                x1 = int(np.clip(x - self.length // 2, 0, w))
                x2 = int(np.clip(x + self.length // 2, 0, w))
                mask[y1: y2, x1: x2] = 0.
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img[k]).cuda()
            # print(mask.shape)
            img[k] = img[k] * mask

        return img
