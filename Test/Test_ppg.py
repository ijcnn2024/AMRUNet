import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Net.models import UNetDS64
import tensorflow as tf
from keras import backend as K
from sklearn.utils import shuffle
from scipy.signal import find_peaks, medfilt


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def normalize(data):
    data = data.astype('float')
    mx = np.max(data, axis=0).astype(np.float64)
    mn = np.min(data, axis=0).astype(np.float64)
    # Workaround to solve the problem of ZeroDivisionError
    return np.true_divide(data - mn, mx - mn, out=np.zeros_like(data - mn), where=(mx - mn) != 0)

def denormalization(ori_data, nor_data):
    data = ori_data.astype('float')
    mx = np.max(data, axis=0).astype(np.float64)
    mn = np.min(data, axis=0).astype(np.float64)
    out = nor_data * (mx - mn) + mn
    return out

def medfilter(data):
    # win_size = int(0.6*125)
    win_size = int(0.7*125)
    base_line = medfilt(data, win_size)
    filter_data = data - base_line
    bias = np.mean(base_line)
    filter_data = filter_data + bias
    return filter_data

def plot_ppg_process(test_path, save_folder):
    save_path = os.path.join(save_folder, test_path.split('/')[-1].replace('.txt', '.png'))
    data = np.loadtxt(test_path)
    ori_ppg = data[:, 0]
    filt_ppg = medfilter(ori_ppg)
    plt.subplots(2, 1, figsize=(12,8))
    plt.subplot(2, 1, 1)
    plt.ylabel("Original PPG")
    plt.plot(ori_ppg)
    plt.subplot(2, 1, 2)
    plt.ylabel("Filtered PPG")
    plt.plot(filt_ppg)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.cla()
    plt.clf()
    plt.close()

def find_period(data):
    peaks, _  = find_peaks(data, distance=50)
    invert_data = np.max(data) - data
    valley, _  = find_peaks(invert_data, distance=50)
    sbp_list, dbp_list = [], []
    for i in range(len(peaks)):
        sbp = data[peaks[i]:]
        dbp = np.min(data[peaks[i]: peaks[i+1]:])
        sbp_list.append(sbp)
        dbp_list.append(dbp)

    plt.plot(data)
    plt.plot(peaks, data[peaks], "^")
    plt.plot(valley, data[valley], "+")

    plt.show()
    return sbp_list, dbp_list

def cal_bp_second(label_bp, pred_bp):
    fs = 125
    label_dbp, label_sbp, label_mbp = [], [], []
    pred_dbp, pred_sbp, pred_mbp = [], [], []
    for i in range(0, len(label_bp)-fs, fs):
        label_bp_segment = label_bp[i: i+fs]
        pred_bp_segment = pred_bp[i: i+fs]
        label_dbp.append(np.max(label_bp_segment))
        label_sbp.append(np.min(label_bp_segment))
        label_mbp.append(np.mean(label_bp_segment))
        pred_dbp.append(np.max(pred_bp_segment))
        pred_sbp.append(np.min(pred_bp_segment))
        pred_mbp.append(np.mean(pred_bp_segment))
    return label_dbp, label_sbp, label_mbp, pred_dbp, pred_sbp, pred_mbp

def LoadModel(model_path):
    model = UNetDS64(length=1024, n_channel=1)
    model.summary()
    model.load_weights(model_path)
    return model

def PlotResult(input, label, output, save_path):
    font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 18,
            }
    sample_rate = 125
    predict_abp = output[0, :, 0]
    x = np.arange(0, (input.shape[1] / sample_rate), (1 / sample_rate))
    fig, axs = plt.subplots(3, 1, figsize=(12,8))

    axs[0].set_title('PPG', font)
    axs[0].plot(x, input[0, :, 0], color='green')
    axs[0].set_ylabel('Amplitude', font)
    axs[0].set_xlabel('Time (s)', font)
    axs[0].legend(['PPG'], loc='upper right', fontsize='small')

    axs[1].set_title('ABP', font)
    axs[1].plot(x, label, color='blue')
    axs[1].plot(x, predict_abp, color='red')
    axs[1].legend(['ABP_label', 'ABP_pred'], loc='upper right', fontsize='small')
    axs[1].set_ylabel('mmHg', font)
    axs[1].set_xlabel('Time (s)', font)

    label_dbp, label_sbp, label_mbp, pred_dbp, pred_sbp, pred_mbp = cal_bp_second(label, predict_abp)
    bp_x_axis = np.arange(1, len(label_dbp)+1, 1)
    axs[2].set_title('ABP_diff', font)
    axs[2].plot(bp_x_axis, label_dbp, marker="^", c='blue', markersize=10)
    axs[2].plot(bp_x_axis, label_sbp, marker="o", c='blue', markersize=10)
    axs[2].plot(bp_x_axis, label_mbp, marker="s", c='blue', markersize=10)
    axs[2].plot(bp_x_axis, pred_dbp, marker="^", c='red', markersize=10)
    axs[2].plot(bp_x_axis, pred_sbp, marker="o", c='red', markersize=10)
    axs[2].plot(bp_x_axis, pred_mbp, marker="s", c='red', markersize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.cla()
    plt.clf()
    plt.close(fig)
    # plt.show()
    # print('finish')

def Test(model, test_path, save_folder):
    data = np.loadtxt(test_path)
    # label = normalize(data[:, 1])
    label = data[:, 1]
    input = np.zeros((1, 1024, 1), dtype=float)
    filt_ppg = medfilter(data[:, 0])
    input[0, :, 0] = normalize(filt_ppg)
    pred_out = model.predict(input)
    # pred_abp = pred_out[0]
    pred_abp = denormalization(label, pred_out[0])
    # find_period(pred_abp[0, :, 0])
    save_path = os.path.join(save_folder, test_path.split('/')[-1].replace('.txt', '.png'))
    # plot ppg, ecg, abp

    PlotResult(input, label, pred_abp, save_path)

if __name__ == '__main__':
    model_path = '/PPGtoABP/Model/PPGUNetDS64/20230323_0132/PPGUNetDS64_Weights_04_0.11403620.hdf5'
    test_csv = '/PPGtoABP/csv/select_data/test.csv'
    save_folder = '/PPGtoABP/Result/PPGUNetDS64/20230323_0132_test'
    create_folder(save_folder)
    model = LoadModel(model_path)
    df_csv = pd.read_csv(test_csv, header=None)
    df_test_1 = shuffle(df_csv)[:100]
    # df_test_1 = df_csv[:100]
    for i in range(len(df_test_1)):
        test_path = df_test_1.iloc[i][0]
        Test(model, test_path, save_folder)
        # plot_ppg_process(test_path, save_folder)
        if i % 10 == 0:
            print(i, 'have been processed!')
