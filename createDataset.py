import os, shutil
from PIL import Image
import numpy as np


path = '~/Datasets/stanford-dogs-dataset/Images/'
dst = '~/ML_Demo/data/'

train_dir = dst + 'train/'
test_dir = dst + 'test/'
val_dir = dst + 'val/'

dir_list = os.listdir(path)
for i in dir_list:
    os.mkdir(train_dir + i.split('-', 1)[1])
    os.mkdir(test_dir + i.split('-', 1)[1])
    os.mkdir(val_dir + i.split('-', 1)[1])
    a = 0
    for j in os.listdir(path+i):
        if a % 2 == 0:
            shutil.copyfile(path+i+'/'+j, train_dir+i.split('-', 1)[1]+'/'+j)
        elif a % 3 == 0:
            shutil.copyfile(path+i+'/'+j, test_dir+i.split('-', 1)[1]+'/'+j)
        else:
            shutil.copyfile(path+i+'/'+j, val_dir+i.split('-', 1)[1]+'/'+j)
        a += 1

