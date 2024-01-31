import os
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def PlotPPG(ppg, save_path):
    font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 20,
            }
    title_list = ['PPG', 'ABP', 'ECG']
    color_list = ['green', 'blue', 'red']
    lead_name = ['PPG', 'ABP', 'ECG']
    sample_rate = 125
    x = np.arange(0, (ppg.shape[0] / sample_rate), (1 / sample_rate))
    # fig, axs = plt.subplots(3, 1, figsize=(12,8))
    fig, axs = plt.subplots(3, 1)
    for i in range(3):
        axs[i].set_title(title_list[i], font)
        axs[i].plot(x, ppg[:, i], color=color_list[i])
        axs[i].set_ylabel('Amplitude', font)
        axs[i].set_xlabel('Time (s)', font)
        axs[i].legend([lead_name[i]], loc='upper right', fontsize='small')


    plt.tight_layout()
    plt.savefig(save_path)
    # plt.savefig(save_path, dpi=300)
    plt.cla()
    plt.clf()
    plt.close(fig)
    # plt.show()
    # print('finish')

def stftPPG(ppg, save_path):
    font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 16,
            }
    sample_rate = 125
    WindowLength = 32
    win = signal.hamming(WindowLength, sym=False)
    Overlap = round(0.75 * WindowLength)

    ppg_signal = ppg[:, 0]
    # 进行stft变换
    f_ppg, t_ppg, Z_ppg = signal.stft(ppg_signal, fs=sample_rate,
                                      window=win,nperseg=WindowLength,
                                      noverlap=Overlap, nfft=256, return_onesided=True)

    ppg_spectrum_real = np.real(Z_ppg).astype('float64')
    ppg_spectrum_imag = np.imag(Z_ppg).astype('float64')

    abp_signal = ppg[:, 1]
    # 进行stft变换
    f_abp, t_abp, Z_abp = signal.stft(abp_signal, fs=sample_rate,
                                      window=win,nperseg=WindowLength,
                                      noverlap=Overlap, nfft=256, return_onesided=True)

    abp_spectrum_real = np.real(Z_abp).astype('float64')
    abp_spectrum_imag = np.imag(Z_abp).astype('float64')

    re_spectrum_ppg = ppg_spectrum_real + 1.0j * ppg_spectrum_imag
    _, ppg_istft = signal.istft(re_spectrum_ppg, fs=sample_rate,  window=win,
                                    nperseg=WindowLength, noverlap=Overlap, nfft=256)
    re_spectrum_abp = abp_spectrum_real + 1.0j * abp_spectrum_imag
    # 进行abp istft变换
    _, abp_istft = signal.istft(re_spectrum_abp, fs=sample_rate,  window=win,
                                    nperseg=WindowLength, noverlap=Overlap, nfft=256)


    x = np.arange(0, (ppg.shape[0] / sample_rate), (1 / sample_rate))
    fig, axs = plt.subplots(4, 1, figsize=(12,16))
    axs[0].set_title('PPG', font)
    axs[0].plot(x, ppg_signal, color='green')
    axs[0].plot(x, ppg_istft, color='red')
    axs[0].legend(['ppg_signal', 'ppg_istft'], loc='upper right', fontsize='small')

    axs[1].set_title('PPG_STFT', font)
    axs[1].pcolormesh(t_ppg, f_ppg, 10*np.log(10*abs(Z_ppg)), cmap='jet')

    axs[2].set_title('ABP', font)
    axs[2].plot(x, abp_signal, color='blue')
    axs[2].plot(x, abp_istft, color='red')
    axs[2].legend(['abp_signal', 'abp_istft'], loc='upper right', fontsize='small')

    axs[3].set_title('ABP_STFT', font)
    axs[3].pcolormesh(t_abp, f_abp, 10*np.log(10*abs(Z_abp)), cmap='jet')

    # plt.tight_layout()
    # plt.savefig(save_path, dpi=300)
    # plt.cla()
    # plt.clf()
    # plt.close(fig)
    plt.show()
    print('finish')
    return ppg_spectrum_real, ppg_spectrum_imag, abp_spectrum_real, abp_spectrum_imag

def istftPPG(ppg_spectrum_real, ppg_spectrum_imag, abp_spectrum_real, abp_spectrum_imag, save_path):
    font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 16,
            }
    sample_rate = 125
    window_length = 256
    fft_num=256
    win = signal.hamming(window_length, sym=False)
    Overlap = round(0.75 * window_length)
    # 进行ppg istft变换
    re_spectrum_ppg = ppg_spectrum_real + 1.0j * ppg_spectrum_imag
    t_ppg, ppg_istft = signal.istft(re_spectrum_ppg, fs=sample_rate,  window=win,
                                            nperseg=window_length, noverlap=Overlap, nfft=fft_num)
    re_spectrum_abp = abp_spectrum_real + 1.0j * abp_spectrum_imag
    # 进行abp istft变换
    t_abp, abp_istft = signal.istft(re_spectrum_abp, fs=sample_rate,  window=win,
                                      nperseg=window_length, noverlap=Overlap, nfft=fft_num)
    x = np.arange(0, (ppg_istft.shape[0] / sample_rate), (1 / sample_rate))
    fig, axs = plt.subplots(2, 1, figsize=(12,4))
    axs[0].set_title('PPG', font)
    axs[0].plot(x, ppg_istft, color='green')
    axs[0].legend(['PPG'], loc='upper right', fontsize='small')

    axs[1].set_title('ABP', font)
    axs[1].plot(x, abp_istft, color='green')
    axs[1].legend(['ABP'], loc='upper right', fontsize='small')
    # plt.tight_layout()
    # plt.savefig(save_path, dpi=300)
    # plt.cla()
    # plt.clf()
    # plt.close(fig)
    plt.show()
    print('finish')
    return

def batch_plotPPG():
    ppg_path = 'C:\\PPGDatabase\\MIMIC_segment\\train'
    save_path = 'C:\\PPGDatabase\\MIMIC_figures\\train'
    ppg_list = os.listdir(ppg_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(12000, 13000):
        ppg = np.loadtxt(ppg_path + '\\' +ppg_list[i])
        PlotPPG(ppg, save_path + '\\' +ppg_list[i].replace('txt', 'png'))


def batch_plotPPG_part2():
    ppg_path = 'C:\\PPGDatabase\\MIMIC_segment\\Part_2'
    save_path = 'C:\\PPGDatabase\\MIMIC_figures\\Part_2'
    ppg_list = os.listdir(ppg_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(6000, 9000):
        ppg = np.loadtxt(ppg_path + '\\' +ppg_list[i])
        PlotPPG(ppg, save_path + '\\' +ppg_list[i].replace('txt', 'png'))


if __name__ == '__main__':

    # batch_plotPPG()
    # batch_plotPPG_part2()
    ppg_path = 'C:\\PPGDatabase\\MIMIC_segment\\train'
    save_path = 'C:\\PPGDatabase\\MIMIC_figures\\train'
    ppg_list = os.listdir(ppg_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    random.shuffle(ppg_list)
    for i in range(10):
        ppg = np.loadtxt(ppg_path + '\\' +ppg_list[i])
        ppg_spectrum_real, ppg_spectrum_imag, abp_spectrum_real, abp_spectrum_imag = stftPPG(ppg, save_path + '\\' +ppg_list[i].replace('txt', 'png') )
        istftPPG(ppg_spectrum_real, ppg_spectrum_imag, abp_spectrum_real, abp_spectrum_imag, save_path + '\\' +ppg_list[i].replace('txt', 'png'))