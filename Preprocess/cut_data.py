import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import resample
import random

def cut_ppg(ori_path, cut_path):
    length = 1024
    ori_list = os.listdir(ori_path)
    for m in range(len(ori_list)):
        count = 0
        print('processing in: ', ori_list[m])
        ori_data = np.loadtxt(os.path.join(ori_path, ori_list[m]))
        for n in range(0, ori_data.shape[0], length):
            # print(count, n)
            # if n+length < ori_data.shape[0] and count < 30:
            if n + length < ori_data.shape[0]:
                segment = ori_data[n:n+length, :]
                np.savetxt(cut_path + '/' + ori_list[m].replace('.txt', '_') + str(count).zfill(5)+'.txt',
                           segment, fmt='%.08f')
                count += 1
            else:
                continue

    print('finish')


if __name__ == '__main__':
    ori_path = '/PPGDatabase/MIMIC/Part_1'
    cut_path = '/PPGDatabase/MIMIC_segment/Part_1_all'
    if not os.path.exists(cut_path):
        os.makedirs(cut_path)
    cut_ppg(ori_path, cut_path)