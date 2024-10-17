# Query-aware Cross-mixup and Cross-reconstruction for Few-shot Fine-grained Image Classification


## Code environment
This code requires Pytorch 1.7.0 and torchvision 0.8.0 or higher with cuda support. It has been tested on Ubuntu 16.04. 

You can create a conda environment with the correct dependencies using the following command lines:
```
conda env create -f environment.yml
```

## Dataset
You must first specify the value of `data_path` in `config.yml`. 

The following datasets are used in our paper: 
- CUB_200_2011 \[[Dataset Page](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)]
- FGVC-Aircraft \[[Dataset Page](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)]
- Flowers \[[Dataset Page](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)]
- Stanford-Cars \[[Paper Page](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6755945)]


The following folders will exist in your `data_path`:
- `CUB_fewshot_cropped`: 100/50/50 classes for train/validation/test, using bounding-box cropped images as input
- `Aircraft_fewshot`: 50/25/25 classes for train/validation/test
- `Flowers`: 51/26/25 classes for train/validation/test
- `Stanford-Cars`: 130/17/49 classes for train/validation/test

## Train
For example, to train QCCN on `CUB_fewshot_cropped` with ResNet-12 as the network backbone, run the following command lines:
```
cd experiments/CUB_fewshot_cropped/QCCN/ResNet-12/
./train.sh
```

## Test
For example, to test QCCN on `CUB_fewshot_cropped` with ResNet-12 as the network backbone under the 5-way 1-shot and  5-way 5-shot setting, run the following command lines:
```
cd experiments/CUB_fewshot_cropped/QCCN/ResNet-12/
python test.py
```



