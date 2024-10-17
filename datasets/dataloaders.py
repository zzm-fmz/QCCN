import codecs
import csv
import json
import os

import torch
import torchvision.datasets as datasets
from PIL import Image

from . import samplers, transform_manager


def get_dataset(data_path,is_training,transform_type,pre):

    dataset = datasets.ImageFolder(
        data_path,
        loader = lambda x: image_loader(path=x,is_training=is_training,transform_type=transform_type,pre=pre))

    return dataset  # image_loader()根据路径对image进行处理



def meta_train_dataloader(data_path,way,shots,transform_type):

    dataset = get_dataset(data_path=data_path,is_training=True,transform_type=transform_type,pre=None)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler = samplers.meta_batchsampler(data_source=dataset,way=way,shots=shots),
        num_workers = 4,
        pin_memory = False)
    return loader



def meta_test_dataloader(data_path,way,shot,pre,transform_type=None,query_shot=16,trial=1000):

    dataset = get_dataset(data_path=data_path,is_training=False,transform_type=transform_type,pre=pre)

    loader = torch.utils.data.DataLoader(
        dataset, # 传入的数据集,有transform操作了
        batch_sampler = samplers.random_sampler(data_source=dataset,way=way,shot=shot,query_shot=query_shot,trial=trial),#
        num_workers = 3,# 决定了有几个进程来处理data loading
        pin_memory = False) # pin_memory:如果设置为True ,data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存
    # batch_sampler 一次只返回一个batch的indices（索引）
    return loader


def normal_train_dataloader(data_path,batch_size,transform_type):

    dataset = get_dataset(data_path=data_path,is_training=True,transform_type=transform_type,pre=None)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 3,
        pin_memory = False,
        drop_last=True)

    return loader


def image_loader(path,is_training,transform_type,pre):

    p = Image.open(path) # 打开图片
    p = p.convert('RGB') # 转换成RGB

    final_transform = transform_manager.get_transform(is_training=is_training,transform_type=transform_type,pre=pre)
    # 定义图像处理的操作

    p = final_transform(p)# 对图片p进行transform操作

    return p


