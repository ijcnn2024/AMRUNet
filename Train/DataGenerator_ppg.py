import os.path

import numpy as np
import keras
import pandas as pd
from scipy.signal import medfilt
import matplotlib.pyplot as plt


class  DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=16, data_len=1024, n_channels=2,
                 data_path='C:\\PPGDatabase\\MIMIC_segment\\train',
                 shuffle=True, ds=False, use_filter=False, error_log="PPGErrorLog.csv"):
        'Initialization'
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.data_len = data_len
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.ds = ds
        self.use_filter=use_filter
        self.error_log = error_log
        self.data_path = data_path
        self.on_epoch_end()

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

    def medfilter(self, data):
        fs = 125
        win_size = int(0.6*fs)
        base_line = medfilt(data, win_size)
        filter_data = data - base_line
        bias = np.mean(base_line)
        filter_data = filter_data + bias
        return filter_data

    def normalize(self, data):
        data = data.astype('float')
        mx = np.max(data, axis=0).astype(np.float64)
        mn = np.min(data, axis=0).astype(np.float64)
        # Workaround to solve the problem of ZeroDivisionError
        return np.true_divide(data - mn, mx - mn, out=np.zeros_like(data - mn), where=(mx - mn) != 0)

    def approximate(self, inp,w_len):
        """
        Downsamples using taking mean over window

        Arguments:
            inp {array} -- signal
            w_len {int} -- length of window

        Returns:
            array -- downsampled signal
        """

        op = []

        for i in range(0,len(inp),w_len):

            op.append(np.mean(inp[i:i+w_len]))

        return np.array(op)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # if self.data_len == 1024 and self.n_channels == 2 :
        #     # Initialization
        #     X = np.zeros((self.batch_size, self.data_len, self.n_channels), dtype=float)
        #     Y = np.zeros((self.batch_size, self.data_len, 1), dtype=float)
        #     # Generate data
        #     for i, ID in enumerate(list_IDs_temp):
        #         try:
        #             # Store sample and labels
        #             # print(os.path.join(self.data_path, ID[0]))
        #             data = np.loadtxt(os.path.join(self.data_path, ID[0])).astype(float)
        #             X[i, :, 0],  X[i, :, 1] = self.normalize(data[:, 0]), self.normalize(data[:, 2])
        #             Y[i, :, 0] = data[:, 1]/200.0
        #         except:
        #             print(ID[0], 'was wrong data')
        #             err_id = pd.DataFrame(ID)
        #             err_id.to_csv(self.error_log, mode='a', index=True, header=False)
        #     return X, Y

        if self.data_len == 1024 and self.n_channels == 1 and self.ds == False:
            # Initialization
            X = np.zeros((self.batch_size, self.data_len, self.n_channels), dtype=float)
            Y = np.zeros((self.batch_size, self.data_len, 1), dtype=float)
            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                try:
                    # Store sample and labels
                    data = np.loadtxt(os.path.join(self.data_path, ID[0])).astype(float)
                    X[i, :, 0] = self.normalize(data[:, 0])
                    Y[i, :, 0] = data[:, 1]/200.0
                except:
                    print(ID[0], 'was wrong data')
                    err_id = pd.DataFrame(ID)
                    err_id.to_csv(self.error_log, mode='a', index=True, header=False)
            return X, Y

        # if self.data_len == 1024 and self.n_channels == 1 and self.ds == True:
        #     # Initialization
        #     X = np.zeros((self.batch_size, self.data_len, self.n_channels), dtype=float)
        #     # Y = np.zeros((5, self.batch_size, self.data_len, 1), dtype=float)
        #     out = np.zeros((self.batch_size, self.data_len, 1), dtype=float)
        #     level1 = np.zeros((self.batch_size, self.data_len//2, 1), dtype=float)
        #     level2= np.zeros((self.batch_size, self.data_len//4, 1), dtype=float)
        #     level3= np.zeros((self.batch_size, self.data_len//8, 1), dtype=float)
        #     level4 = np.zeros((self.batch_size, self.data_len//16, 1), dtype=float)
        #     Y = [out, level1, level2, level3, level4]
        #
        #
        #     # Generate data
        #     for i, ID in enumerate(list_IDs_temp):
        #         try:
        #             # Store sample and labels
        #             data = np.loadtxt(os.path.join(self.data_path, ID[0])).astype(float)
        #             X[i, :, 0] = self.normalize(data[:, 0])
        #             label = self.normalize(data[:, 1])
        #             Y[0][i, :, 0] = label
        #             Y[1][i, :, 0]= self.approximate(label, 2)
        #             Y[2][i, :, 0]= self.approximate(label, 4)
        #             Y[3][i, :, 0]= self.approximate(label, 8)
        #             Y[4][i, :, 0]= self.approximate(label, 16)
        #
        #         except:
        #             print(ID[0], 'was wrong data')
        #             err_id = pd.DataFrame(ID)
        #             err_id.to_csv(self.error_log, mode='a', index=True, header=False)
        #     return X, Y

        if self.data_len == 1024 and self.n_channels == 1 and self.ds == True and self.use_filter == True:
            # Initialization
            X = np.zeros((self.batch_size, self.data_len, self.n_channels), dtype=float)
            # Y = np.zeros((5, self.batch_size, self.data_len, 1), dtype=float)
            out = np.zeros((self.batch_size, self.data_len, 1), dtype=float)
            level1 = np.zeros((self.batch_size, self.data_len//2, 1), dtype=float)
            level2= np.zeros((self.batch_size, self.data_len//4, 1), dtype=float)
            level3= np.zeros((self.batch_size, self.data_len//8, 1), dtype=float)
            level4 = np.zeros((self.batch_size, self.data_len//16, 1), dtype=float)
            Y = [out, level1, level2, level3, level4]


            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                try:
                    # Store sample and labels
                    data = np.loadtxt(os.path.join(self.data_path, ID[0])).astype(float)
                    filt_ppg = self.medfilter(data[:, 0])
                    # plt.subplots(2, 1, figsize=(12,8))
                    # plt.subplot(2, 1, 1)
                    # plt.ylabel("Original PPG")
                    # plt.plot(data[:, 0])
                    # plt.subplot(2, 1, 2)
                    # plt.ylabel("Filtered PPG")
                    # plt.plot(filt_ppg)
                    # plt.tight_layout()
                    # plt.pause(2)
                    X[i, :, 0] = self.normalize(filt_ppg)
                    label = self.normalize(data[:, 1])
                    Y[0][i, :, 0] = label
                    Y[1][i, :, 0]= self.approximate(label, 2)
                    Y[2][i, :, 0]= self.approximate(label, 4)
                    Y[3][i, :, 0]= self.approximate(label, 8)
                    Y[4][i, :, 0]= self.approximate(label, 16)

                except:
                    print(ID[0], 'was wrong data')
                    err_id = pd.DataFrame(ID)
                    err_id.to_csv(self.error_log, mode='a', index=True, header=False)
            return X, Y

        if self.data_len == 1024 and self.n_channels == 2 and self.ds == True and self.use_filter == True:
            # Initialization
            X = np.zeros((self.batch_size, self.data_len, self.n_channels), dtype=float)
            # Y = np.zeros((5, self.batch_size, self.data_len, 1), dtype=float)
            out = np.zeros((self.batch_size, self.data_len, 1), dtype=float)
            level1 = np.zeros((self.batch_size, self.data_len//2, 1), dtype=float)
            level2= np.zeros((self.batch_size, self.data_len//4, 1), dtype=float)
            level3= np.zeros((self.batch_size, self.data_len//8, 1), dtype=float)
            level4 = np.zeros((self.batch_size, self.data_len//16, 1), dtype=float)
            Y = [out, level1, level2, level3, level4]


            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                try:
                    # Store sample and labels
                    data = np.loadtxt(os.path.join(self.data_path, ID[0])).astype(float)
                    filt_ppg = self.medfilter(data[:, 0])
                    filt_ecg = self.medfilter(data[:, 2])
                    # plt.subplots(4, 1, figsize=(16,8))
                    # plt.subplot(4, 1, 1)
                    # plt.ylabel("Original PPG")
                    # plt.plot(data[:, 0])
                    # plt.subplot(4, 1, 2)
                    # plt.ylabel("Filtered PPG")
                    # plt.plot(filt_ppg)
                    # plt.subplot(4, 1, 3)
                    # plt.ylabel("Original ECG")
                    # plt.plot(data[:, 2])
                    # plt.subplot(4, 1, 4)
                    # plt.ylabel("Filtered ECG")
                    # plt.plot(filt_ecg)
                    # plt.tight_layout()
                    # plt.pause(2)
                    X[i, :, 0] = self.normalize(filt_ppg)
                    X[i, :, 1] = self.normalize(filt_ecg)
                    label = self.normalize(data[:, 1])
                    Y[0][i, :, 0] = label
                    Y[1][i, :, 0]= self.approximate(label, 2)
                    Y[2][i, :, 0]= self.approximate(label, 4)
                    Y[3][i, :, 0]= self.approximate(label, 8)
                    Y[4][i, :, 0]= self.approximate(label, 16)

                except:
                    print(ID[0], 'was wrong data')
                    err_id = pd.DataFrame(ID)
                    err_id.to_csv(self.error_log, mode='a', index=True, header=False)
            return X, Y
