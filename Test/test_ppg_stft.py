import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from sklearn.utils import shuffle
from Net.PPG_DCN import PPGModel
from scipy import signal


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

def LoadModel(model_path):
    model = PPGModel()
    model.summary()
    model.load_weights(model_path)
    return model

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


def PlotResult(input, label, output, save_path):
    font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 18,
            }
    sample_rate = 125
    x = np.arange(0, (input.shape[0] / sample_rate), (1 / sample_rate))
    fig, axs = plt.subplots(3, 1, figsize=(12,8))
    aaaa = input[0, :]
    axs[0].set_title('PPG', font)
    axs[0].plot(x, input[:, 0], color='green')
    axs[0].set_ylabel('Amplitude', font)
    axs[0].set_xlabel('Time (s)', font)
    axs[0].legend(['PPG'], loc='upper right', fontsize='small')

    axs[1].set_title('ABP', font)
    axs[1].plot(x, label, color='blue')
    axs[1].plot(x, output, color='red')
    axs[1].legend(['ABP_label', 'ABP_pred'], loc='upper right', fontsize='small')
    axs[1].set_ylabel('mmHg', font)
    axs[1].set_xlabel('Time (s)', font)

    axs[2].set_title('ABP_diff', font)
    axs[2].plot(x, output - label, color='black')
    axs[2].set_ylabel('mmHg', font)
    axs[2].set_xlabel('Time (s)', font)
    axs[2].legend(['ABP_diff'], loc='upper right', fontsize='small')



    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.cla()
    plt.clf()
    plt.close(fig)
    # plt.show()
    # print('finish')

def Test(model, test_path, save_folder):
    data = np.loadtxt(test_path)
    label = data[:, 1]
    input = np.zeros((1, 129, 129, 2), dtype=float)
    input[0, : , :, 0], input[0, : , :, 1] = stftPPG(data)
    pred_out = model.predict(input)
    pred_abp = istft(pred_out)
    save_path = os.path.join(save_folder, test_path.split('\\')[-1].replace('.txt', '.png'))
    # plot ppg, ecg, abp
    PlotResult(data, label, pred_abp, save_path)


if __name__ == '__main__':
    model_path = 'C:\\ppg\\Model\\PPGDCN\\20230103_1833\\PPGDCN_Weights_06_6.53350300.hdf5'
    test_csv = 'C:\\ppg\\csv\\MIMIC\\val.csv'
    data_folder = 'C:\\PPGDatabase\\MIMIC_segment\\val'
    save_folder = 'C:\\ppg\\Result\\20230103_1833_val'
    create_folder(save_folder)
    model = LoadModel(model_path)
    df_csv = pd.read_csv(test_csv, header=None)
    df_test_1 = shuffle(df_csv)[:100]
    for i in range(len(df_test_1)):
        test_path = os.path.join(data_folder, df_csv.iloc[i][0])
        Test(model, test_path, save_folder)
        if i % 10 == 0:
            print(i, 'have been processed!')
