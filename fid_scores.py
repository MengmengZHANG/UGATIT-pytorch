# example of calculating the frechet inception distance
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm

import torch.utils.data as data

from PIL import Image

import os
import os.path
import time, itertools
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *
from glob import glob

# calculate frechet inception distance
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

# define two collections of activations
# act1 = random(10*2048)
# act1 = act1.reshape((10,2048))
# act2 = random(10*2048)
# act2 = act2.reshape((10,2048))
# # fid between act1 and act1
# print ('sample act1, act2:', act1.shape, act2.shape)
# fid = calculate_fid(act1, act1)
# print('FID (same): %.3f' % fid)
# # fid between act1 and act2
# fid = calculate_fid(act1, act2)
# print('FID (different): %.3f' % fid)

# load data
dataset = 'selfie2anime_64_64'
batch_size = 1
img_size = 64
train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((img_size + 30, img_size+30)),
            transforms.RandomCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

trainA = ImageFolder(os.path.join('dataset', dataset, 'trainA'), train_transform)
trainB = ImageFolder(os.path.join('dataset', dataset, 'trainB'), train_transform)
trainA_loader = DataLoader(trainA, batch_size=batch_size, shuffle=False)
trainB_loader = DataLoader(trainB, batch_size=batch_size, shuffle=False)

#sample = trainA_loader[0]
last = None
for idx, item in enumerate(trainA_loader):
    item = item[0]
    item = item.view(64, -1)
    item = numpy.array(item)
    print (item.shape, type(item))
    if idx != 0:
        fid = calculate_fid(item, last)
        print ('fid diff:', fid)
    last = item