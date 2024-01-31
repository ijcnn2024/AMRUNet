import numpy as np
import keras
import pandas as pd
import os
from scipy import signal


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=16, f_len=129, t_len=129, n_dims=2,
                 sample_rate=125, WindowLength=32, Overlap=0.75, nfft=256,
                 data_path='C:\\PPGDatabase\\MIMIC_segment\\train',
                 shuffle=True, error_log="ppgErrorLog.csv"):
        'Initialization'
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.f_len = f_len
        self.t_len = t_len
        self.n_dims = n_dims
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.WindowLength = WindowLength
        self.win = signal.hamming(WindowLength, sym=False)
        self.Overlap = round(Overlap * WindowLength)
        self.nfft = nfft
        self.data_path = data_path
        self.error_log = error_log
        self.on_epoch_end()
        # self.__readcsv()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __normalize(self, data):
        data = data.astype('float')
        mx = np.max(data, axis=0).astype(np.float64)
        mn = np.min(data, axis=0).astype(np.float64)
        # Workaround to solve the problem of ZeroDivisionError
        return np.true_divide(data - mn, mx - mn, out=np.zeros_like(data - mn), where=(mx - mn) != 0)


    def __stft(self, data):
        f, t, Z = signal.stft(data,
                              fs=self.sample_rate,
                              window=self.win,
                              nperseg=self.WindowLength,
                              noverlap=self.Overlap,
                              nfft=self.nfft,
                              return_onesided=True)
        aaa = np.real(Z).astype('float64')
        bbb = np.imag(Z).astype('float64')
        return np.real(Z).astype('float64'), np.imag(Z).astype('float64')


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        if self.f_len == 129 and self.t_len == 129 and self.n_dims== 2 :
            # Initialization
            X = np.empty((self.batch_size, self.f_len, self.t_len, self.n_dims), dtype=float)
            Y = np.empty((self.batch_size, self.f_len, self.t_len, self.n_dims), dtype=float)
            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                data = np.loadtxt(ID[0]).astype(float)
                X[i, :, :, 0], X[i, :, :, 1] = self.__stft(data[:, 0])

                # Store labels
                Y[i, :, :, 0],  Y[i, :, :, 1] = self.__stft(data[:, 1])

            return X, Y


