import os
import numpy as np
import matplotlib.pyplot as plt
from Net.models import AMRUNetBP
import tensorflow as tf
from keras import backend as K
from scipy import signal
from scipy.signal import find_peaks, medfilt
from scipy.stats import pearsonr
import shutil

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        }

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

def stftPPG(data):
    sample_rate = 125
    WindowLength = 32
    win = signal.hamming(WindowLength, sym=False)
    Overlap = round(0.75 * WindowLength)
    ppg_signal = data[:, 0]
    # 进行stft变换
    f_ppg, t_ppg, Z_ppg = signal.stft(ppg_signal, fs=sample_rate,
                                      window=win,nperseg=WindowLength,
                                      noverlap=Overlap, nfft=256, return_onesided=True)

    ppg_spectrum_real = np.real(Z_ppg).astype('float64')
    ppg_spectrum_imag = np.imag(Z_ppg).astype('float64')
    return ppg_spectrum_real, ppg_spectrum_imag

def istft(feature_pred):
    sample_rate = 125
    window_length = 32
    fft_num=256
    win = signal.hamming(window_length, sym=False)
    Overlap = round(0.75 * window_length)

    feature_pred_real = feature_pred[0, :, :, 0]
    feature_pred_imag = feature_pred[0, :, :, 1]
    re_spectrum = feature_pred_real + 1.0j * feature_pred_imag
    # 进行istft变换
    _, abp_istft = signal.istft(re_spectrum, fs=sample_rate,  window=win,
                                nperseg=window_length, noverlap=Overlap, nfft=fft_num)
    return abp_istft

def plot_ppg_process(test_path, save_folder):
    save_path = os.path.join(save_folder, test_path.split('\\')[-1].replace('.txt', '.png'))
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
        sbp = data[peaks[i]]
        # dbp = np.min(data[peaks[i]: peaks[i+1]])
        dbp = data[valley[i]]
        sbp_list.append(sbp)
        dbp_list.append(dbp)

    # plt.plot(data)
    # plt.plot(peaks, data[peaks], "^")
    # plt.plot(valley, data[valley], "+")

    # plt.show()
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

def cal_bp_peaks(label_bp, pred_bp):
    label_sbp_list, label_dbp_list = find_period(label_bp)
    pred_sbp_list, pred_dbp_list = find_period(pred_bp)


    label_dbp, label_sbp, label_mbp = [], [], []
    pred_dbp, pred_sbp, pred_mbp = [], [], []
    if len(label_sbp_list) == len(label_dbp_list):
        for i in range(len(label_sbp_list)-5):
            label_sbp.append(np.max(label_sbp_list[i:i+5]))
            label_dbp.append(np.min(label_dbp_list[i:i+5]))
        for j in  range(len(label_sbp)):
            label_mbp.append((label_sbp[j] + label_dbp[j])/2)

    if len(pred_sbp_list) == len(pred_dbp_list):
        for i in range(len(pred_sbp_list)-5):
            pred_sbp.append(np.max(pred_sbp_list[i:i+5]))
            pred_dbp.append(np.min(pred_dbp_list[i:i+5]))
        for j in range(len(pred_sbp)):
            pred_mbp.append((pred_sbp[j] + pred_dbp[j])/2)

    return label_dbp, label_sbp, label_mbp, pred_dbp, pred_sbp, pred_mbp

def cal_bp_diff(label_dbp, label_sbp, label_mbp, pred_dbp, pred_sbp, pred_mbp):
    dbp_diff, sbp_diff, mbp_diff = [], [], []
    for i in range(len(label_dbp)):
        dbp_diff.append(label_dbp[i] - pred_dbp[i])
        sbp_diff.append(label_sbp[i] - pred_sbp[i])
        mbp_diff.append(label_mbp[i] - pred_mbp[i])

    return np.abs(np.round(dbp_diff, 4)), np.abs(np.round(sbp_diff, 4)), np.abs(np.round(mbp_diff, 4))

def LoadModel(model_path):
    model = AMRUNetBP(length=1024, n_channel=2)
    model.summary()
    model.load_weights(model_path)
    return model

def PlotResult(input, label, output, test_path, select_path, under_select_path):

    sample_rate = 125
    predict_abp = output[0, :, 0]
    label_dbp, label_sbp, label_mbp, pred_dbp, pred_sbp, pred_mbp = cal_bp_second(label, predict_abp)
    corrcoef, pvalve = pearsonr(label, predict_abp)
    dbp_diff, sbp_diff, mbp_diff = cal_bp_diff(label_dbp, label_sbp, label_mbp, pred_dbp, pred_sbp, pred_mbp)
    # if np.max(dbp_diff) > 6 or np.max(sbp_diff) > 6 or np.max(mbp_diff) > 6 or corrcoef < 0.8:
    if np.max(dbp_diff) > 6 or np.max(sbp_diff) > 6 or corrcoef < 0.8:

        x = np.arange(0, (input.shape[0] / sample_rate), (1 / sample_rate))
        fig, axs = plt.subplots(3, 1, figsize=(12,8))

        axs[0].set_title('PPG', font)
        axs[0].plot(x, input[:, 0], color='green')
        axs[0].set_ylabel('Amplitude', font)
        axs[0].set_xlabel('Time (s)', font)
        axs[0].legend(['PPG'], loc='upper right', fontsize='small')

        axs[1].set_title('ABP', font)
        axs[1].plot(x, label, color='blue')
        axs[1].plot(x, predict_abp, color='red')
        axs[1].legend(['ABP_label', 'ABP_pred'], loc='upper right', fontsize='small')
        axs[1].set_ylabel('mmHg', font)
        axs[1].set_xlabel('Time (s)', font)

        bp_x_axis = np.arange(1, len(label_dbp)+1, 1)
        axs[2].set_title('ABP_diff' + '  ' + str(np.round(corrcoef, 4)) + '  '+ str(np.round(pvalve, 4)) + '  \n'
                         + str(dbp_diff) + '  \n'+ str(sbp_diff) + '  \n' + str(mbp_diff))
        axs[2].plot(bp_x_axis, label_dbp, marker="^", c='blue', markersize=10)
        axs[2].plot(bp_x_axis, label_sbp, marker="o", c='blue', markersize=10)
        axs[2].plot(bp_x_axis, label_mbp, marker="s", c='blue', markersize=10)
        axs[2].plot(bp_x_axis, pred_dbp, marker="^", c='red', markersize=10)
        axs[2].plot(bp_x_axis, pred_sbp, marker="o", c='red', markersize=10)
        axs[2].plot(bp_x_axis, pred_mbp, marker="s", c='red', markersize=10)

        plt.tight_layout()
        plt.savefig(under_select_path, dpi=300)
        plt.cla()
        plt.clf()
        plt.close(fig)
        # plt.show()
        # print('finish')
    else:
        # shutil.copy(test_path, select_path)
        x = np.arange(0, (input.shape[0] / sample_rate), (1 / sample_rate))
        fig, axs = plt.subplots(3, 1, figsize=(12,8))

        axs[0].set_title('PPG', font)
        axs[0].plot(x, input[:, 0], color='green')
        axs[0].set_ylabel('Amplitude', font)
        axs[0].set_xlabel('Time (s)', font)
        axs[0].legend(['PPG'], loc='upper right', fontsize='small')

        axs[1].set_title('ABP', font)
        axs[1].plot(x, label, color='blue')
        axs[1].plot(x, predict_abp, color='red')
        axs[1].legend(['ABP_label', 'ABP_pred'], loc='upper right', fontsize='small')
        axs[1].set_ylabel('mmHg', font)
        axs[1].set_xlabel('Time (s)', font)

        bp_x_axis = np.arange(1, len(label_dbp)+1, 1)
        axs[2].set_title('ABP_diff' + '  ' + str(np.round(corrcoef, 4)) + '  '+ str(np.round(pvalve, 4)) + '  \n'
                         + str(dbp_diff) + '  \n'+ str(sbp_diff) + '  \n' + str(mbp_diff))
        axs[2].plot(bp_x_axis, label_dbp, marker="^", c='blue', markersize=10)
        axs[2].plot(bp_x_axis, label_sbp, marker="o", c='blue', markersize=10)
        axs[2].plot(bp_x_axis, label_mbp, marker="s", c='blue', markersize=10)
        axs[2].plot(bp_x_axis, pred_dbp, marker="^", c='red', markersize=10)
        axs[2].plot(bp_x_axis, pred_sbp, marker="o", c='red', markersize=10)
        axs[2].plot(bp_x_axis, pred_mbp, marker="s", c='red', markersize=10)

        plt.tight_layout()
        plt.savefig(select_path.replace('.txt', '.png'), dpi=300)
        plt.cla()
        plt.clf()
        plt.close(fig)
        # plt.show()
        # print('finish')

def PlotResult_peaks(input, label, output, test_path, select_path, under_select_path):
    sample_rate = 125
    predict_abp = output[0, :, 0]
    label_dbp, label_sbp, label_mbp, pred_dbp, pred_sbp, pred_mbp = cal_bp_peaks(label, predict_abp)
    corrcoef, pvalve = pearsonr(label, predict_abp)
    dbp_diff, sbp_diff, mbp_diff = cal_bp_diff(label_dbp, label_sbp, label_mbp, pred_dbp, pred_sbp, pred_mbp)
    if np.max(dbp_diff) > 6 or np.max(sbp_diff) > 6 or corrcoef < 0.8:

        x = np.arange(0, (input.shape[0] / sample_rate), (1 / sample_rate))
        fig, axs = plt.subplots(3, 1, figsize=(12,8))

        axs[0].set_title('PPG', font)
        axs[0].plot(x, input[:, 0], color='green')
        axs[0].set_ylabel('Amplitude', font)
        axs[0].set_xlabel('Time (s)', font)
        axs[0].legend(['PPG'], loc='upper right', fontsize='small')

        axs[1].set_title('ABP', font)
        axs[1].plot(x, label, color='blue')
        axs[1].plot(x, predict_abp, color='red')
        axs[1].legend(['ABP_label', 'ABP_pred'], loc='upper right', fontsize='small')
        axs[1].set_ylabel('mmHg', font)
        axs[1].set_xlabel('Time (s)', font)

        bp_x_axis = np.arange(1, len(label_dbp)+1, 1)
        axs[2].set_title('ABP_diff' + '  ' + str(np.round(corrcoef, 4)) + '  '+ str(np.round(pvalve, 4)) + '  \n'
                         + str(dbp_diff) + '  \n'+ str(sbp_diff) + '  \n' + str(mbp_diff))
        axs[2].plot(bp_x_axis, label_dbp, marker="^", c='blue', markersize=10)
        axs[2].plot(bp_x_axis, label_sbp, marker="o", c='blue', markersize=10)
        axs[2].plot(bp_x_axis, label_mbp, marker="s", c='blue', markersize=10)
        axs[2].plot(bp_x_axis, pred_dbp, marker="^", c='red', markersize=10)
        axs[2].plot(bp_x_axis, pred_sbp, marker="o", c='red', markersize=10)
        axs[2].plot(bp_x_axis, pred_mbp, marker="s", c='red', markersize=10)

        plt.tight_layout()
        plt.savefig(under_select_path, dpi=300)
        plt.cla()
        plt.clf()
        plt.close(fig)
        # plt.show()
        # print('finish')
    else:
        shutil.copy(test_path, select_path)
        x = np.arange(0, (input.shape[0] / sample_rate), (1 / sample_rate))
        fig, axs = plt.subplots(3, 1, figsize=(12,8))

        axs[0].set_title('PPG', font)
        axs[0].plot(x, input[:, 0], color='green')
        axs[0].set_ylabel('Amplitude', font)
        axs[0].set_xlabel('Time (s)', font)
        axs[0].legend(['PPG'], loc='upper right', fontsize='small')

        axs[1].set_title('ABP', font)
        axs[1].plot(x, label, color='blue')
        axs[1].plot(x, predict_abp, color='red')
        axs[1].legend(['ABP_label', 'ABP_pred'], loc='upper right', fontsize='small')
        axs[1].set_ylabel('mmHg', font)
        axs[1].set_xlabel('Time (s)', font)

        bp_x_axis = np.arange(1, len(label_dbp)+1, 1)
        axs[2].set_title('ABP_diff' + '  ' + str(np.round(corrcoef, 4)) + '  '+ str(np.round(pvalve, 4)) + '  \n'
                         + str(dbp_diff) + '  \n'+ str(sbp_diff) + '  \n' + str(mbp_diff))
        axs[2].plot(bp_x_axis, label_dbp, marker="^", c='blue', markersize=10)
        axs[2].plot(bp_x_axis, label_sbp, marker="o", c='blue', markersize=10)
        axs[2].plot(bp_x_axis, label_mbp, marker="s", c='blue', markersize=10)
        axs[2].plot(bp_x_axis, pred_dbp, marker="^", c='red', markersize=10)
        axs[2].plot(bp_x_axis, pred_sbp, marker="o", c='red', markersize=10)
        axs[2].plot(bp_x_axis, pred_mbp, marker="s", c='red', markersize=10)

        plt.tight_layout()
        plt.savefig(select_path.replace('.txt', '.png'), dpi=300)
        plt.cla()
        plt.clf()
        plt.close(fig)
        # plt.show()
        # print('finish')

def Test_peaks_stft():
    model_path = 'C:\\ppg\\Model\\PPGECGUNetDS64\\20230203_1715\\PPGECGUNetDS64.hdf5'
    ppg_path = 'C:\\PPGDatabase\\MIMIC_segment\\Part_2'
    save_folder = 'C:\\ppg\\Result\\PPGDCN\\20230301_1715_test'
    ppg_list = os.listdir(ppg_path)
    create_folder(save_folder)
    model = LoadModel(model_path)

    for i in range(10100, 11000):
        test_path = os.path.join(ppg_path, ppg_list[i])
        data = np.loadtxt(test_path)
        label = data[:, 1]
        input = np.zeros((1, 129, 129, 2), dtype=float)
        input[0, : , :, 0], input[0, : , :, 1] = stftPPG(data)
        pred_out = model.predict(input)
        pred_abp = istft(pred_out)
        save_path = os.path.join(save_folder, test_path.split('\\')[-1].replace('.txt', '.png'))
        # plot ppg, ecg, abp
        PlotResult(data, label, pred_abp, save_path)

        if i % 100 == 0:
            print(i, 'have been processed!')

def Test_peaks():
    model_path = 'C:\\ppg\\Model\\PPGECGUNetDS64\\20230203_1715\\PPGECGUNetDS64.hdf5'
    ppg_path = 'C:\\PPGDatabase\\MIMIC_segment\\Part_2'
    select_folder = 'C:\\PPGDatabase\\MIMIC_figures\\Part_2_select_100test'
    under_select_folder = 'C:\\PPGDatabase\\MIMIC_figures\\Part_2_under_select_path_100test'
    ppg_list = os.listdir(ppg_path)
    create_folder(select_folder)
    create_folder(under_select_folder)
    model = LoadModel(model_path)

    for i in range(11000, 11100):
        test_path = os.path.join(ppg_path, ppg_list[i])
        data = np.loadtxt(test_path)
        label = data[:, 1]
        input = np.zeros((1, 1024, 2), dtype=float)
        filt_ppg = medfilter(data[:, 0])
        filt_ecg = medfilter(data[:, 2])
        input[0, :, 0] = normalize(filt_ppg)
        input[0, :, 1] = normalize(filt_ecg)
        pred_out = model.predict(input)
        # pred_abp = pred_out[0]
        pred_abp = denormalization(label, pred_out[0])


        # corrcoef, pvalve = pearsonr(label, pred_abp[0, :, 0])
        # dbp_diff, sbp_diff, mbp_diff = cal_bp_diff(label, pred_abp)
        # if dbp_diff.any() > 6 or sbp_diff.any() > 6 or mbp_diff > 6:
        #     print(test_path)

        # find_period(pred_abp[0, :, 0])
        select_path = os.path.join(select_folder, test_path.split('\\')[-1])
        under_select_path = os.path.join(under_select_folder, test_path.split('\\')[-1].replace('.txt', '.png'))
        # plot ppg, ecg, abp
        PlotResult_peaks(data, label, pred_abp, test_path, select_path, under_select_path)

        if i % 100 == 0:
            print(i, 'have been processed!')


def Test(model, test_path, save_folder):
    data = np.loadtxt(test_path)
    # label = normalize(data[:, 1])
    label = data[:, 1]
    input = np.zeros((1, 1024, 2), dtype=float)
    filt_ppg = medfilter(data[:, 0])
    filt_ecg = medfilter(data[:, 2])
    input[0, :, 0] = normalize(filt_ppg)
    input[0, :, 1] = normalize(filt_ecg)
    pred_out = model.predict(input)
    # pred_abp = pred_out[0]
    pred_abp = denormalization(label, pred_out[0])
    # find_period(pred_abp[0, :, 0])
    save_path = os.path.join(save_folder, test_path.split('\\')[-1].replace('.txt', '.png'))
    # plot ppg, ecg, abp

    PlotResult(input, label, pred_abp, save_path)

def Select(model, test_path, select_folder, under_select_folder):
    data = np.loadtxt(test_path)
    label = data[:, 1]
    input = np.zeros((1, 1024, 2), dtype=float)
    filt_ppg = medfilter(data[:, 0])
    filt_ecg = medfilter(data[:, 2])
    input[0, :, 0] = normalize(filt_ppg)
    input[0, :, 1] = normalize(filt_ecg)
    pred_out = model.predict(input)
    # pred_abp = pred_out[0]
    pred_abp = denormalization(label, pred_out[0])


    # corrcoef, pvalve = pearsonr(label, pred_abp[0, :, 0])
    # dbp_diff, sbp_diff, mbp_diff = cal_bp_diff(label, pred_abp)
    # if dbp_diff.any() > 6 or sbp_diff.any() > 6 or mbp_diff > 6:
    #     print(test_path)

    # find_period(pred_abp[0, :, 0])
    select_path = os.path.join(select_folder, test_path.split('\\')[-1])
    under_select_path = os.path.join(under_select_folder, test_path.split('\\')[-1].replace('.txt', '.png'))
    # plot ppg, ecg, abp
    PlotResult(data, label, pred_abp, test_path, select_path, under_select_path)


if __name__ == '__main__':
    Test_peaks()

