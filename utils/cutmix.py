import numpy as np
import torch


class Cutmix(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, beta, cutmix_prob, way, shot, query_shot):
        self.beta = beta
        self.cutmix_prob=cutmix_prob
        self.way = way
        self.shot = shot
        self.query_shot = query_shot


    def rand_bbox(self,size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (B, C, H, W). [50, 3, 84, 84]
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        r = np.random.rand(1)
        global support
        support = img[: self.way * self.shot]
        if self.beta > 0 and r < self.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)


            bbx1, bby1, bbx2, bby2 = self.rand_bbox(support.size(), lam)

            select_query = []  # list
            for i in range(self.way):
                # select_range = range(self.way * self.shot + i * self.query_shot, (self.way * self.shot + self.query_shot * i) + self.query_shot)
                # select_id = random.sample(select_range, self.shot)
                select_id = torch.randint(self.way * self.shot + i * self.query_shot,
                                          (self.way * self.shot + self.query_shot * i) + self.query_shot,
                                          [self.shot]).data.numpy()
                [select_query.append(img[k]) for k in select_id]

            select_query = torch.tensor([item.cpu().detach().numpy() for item in select_query]).cuda()  # 每次总共50个抽出来的query进行相乘

            support[:, :, bbx1:bbx2, bby1:bby2] = select_query[:, :, bbx1:bbx2, bby1:bby2]

            return support
        else:
            return support
