import os
import pdb
import random

import numpy as np
import scipy
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


def mkdir(path):
    if os.path.exists(path):
        print("---  the folder already exists  ---")
    else:
        os.makedirs(path)


# get pre-resized 84x84 images for validation and test
def get_pre_folder(image_folder, transform_type):
    split = ['val', 'test']

    if transform_type == 0:
        transform = transforms.Compose([transforms.Resize(92),
                                        transforms.CenterCrop(84)])
    elif transform_type == 1:
        transform = transforms.Compose([transforms.Resize([92, 92]),
                                        transforms.CenterCrop(84)])

    cat_list = []

    for i in split:

        cls_list = os.listdir(os.path.join(image_folder, i))

        folder_name = i + '_pre'

        mkdir(os.path.join(image_folder, folder_name))

        for j in tqdm(cls_list):

            mkdir(os.path.join(image_folder, folder_name, j))

            img_list = os.listdir(os.path.join(image_folder, i, j))

            for img_name in img_list:
                img = Image.open(os.path.join(image_folder, i, j, img_name))
                img = img.convert('RGB')
                img = transform(img)
                img.save(os.path.join(image_folder, folder_name, j, img_name[:-3] + 'png'))


def get_device_map(gpu):
    cuda = lambda x: 'cuda:%d' % x
    temp = {}
    for i in range(4):
        temp[cuda(i)] = cuda(gpu)
    return temp




def mixup_data(x, way, shot, query_shot, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    support = x[:way * shot]  # torch.Size([50, 3, 84, 84])
    select_query = []  # list
    for i in range(way):
        select_id = torch.randint(way * shot + i * query_shot, (way * shot + query_shot * i) + query_shot,[shot]).data.numpy()
        [select_query.append(x[k]) for k in select_id]

    select_query = torch.tensor([item.cpu().detach().numpy() for item in select_query]).cuda()  
    mixed_s = lam * support + (1 - lam) * select_query
    return mixed_s




def mixup_data_r18(x, way, shot, query_shot, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = []
        for i in range(way * shot):
            lam.append(np.random.beta(alpha, alpha))
        lam = np.array(lam)
        lam = torch.tensor(np.expand_dims(lam, 1)).cuda()
    else:
        lam = 1
    support = x[:way * shot]  # torch.Size([50, 3, 84, 84])
    b, c, h, w = support.size()
    select_query = []  # list
    for i in range(way):
        select_id = torch.randint(way * shot + i * query_shot, (way * shot + query_shot * i) + query_shot,[shot]).data.numpy()
        [select_query.append(x[k]) for k in select_id]

    select_query = torch.tensor([item.cpu().detach().numpy() for item in select_query]).cuda()  
    mixed_s = (support.view(b, -1) * lam + select_query.view(b, -1) * (1 - lam)).view(b, c, h, w).float()

    return mixed_s


