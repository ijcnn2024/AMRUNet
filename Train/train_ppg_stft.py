"""
ppg data samplerate 125, segment_data len 1024 for training
"""

import os
import datetime
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import math
import keras.backend as K
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from Net.DataGenerator_PPG_DCN import DataGenerator
from Net.PPG_DCN import PPGModel_plus



def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_csv(csv_path):
    out_list = []
    df_csv = pd.read_csv(csv_path, header=None)
    for i in range(len(df_csv)):
        out_list.append(df_csv.iloc[i, :])
    return out_list



def main():
    params = {'batch_size': 8,
              'data_path': 'C:\\PPGDatabase\\MIMIC_segment\\train',
              'shuffle': True,
              'error_log': 'PPGErrorLog.csv'}

    paths = {'model_name': 'PPGDCN',
             'model_dir': 'C:\\ppg\\Model\\PPGDCN',
             'log_dir': 'C:\\ppg\\Logs\\PPGDCN',
             'train_csv': 'C:\\ppg\\csv\\select_data\\train.csv',
             'val_csv': 'C:\\ppg\\csv\\select_data\\val.csv',
             'model_weight': 'PPGDCN.hdf5',
             'train_date': datetime.datetime.now().strftime("%Y%m%d_%H%M"),
             'select_date': ''
             }
    create_folder(paths['model_dir'] + os.sep + paths['train_date'])
    create_folder(paths['log_dir'] + os.sep + paths['train_date'])

    train_list = read_csv(paths['train_csv'])
    val_list = read_csv(paths['val_csv'])
    partition = {'train': train_list, 'val':val_list}

    train_generator = DataGenerator(partition['train'], **params)
    params['shuffle'] = False
    params['data_path'] = 'C:\\PPGDatabase\\MIMIC_segment\\val'
    val_generator = DataGenerator(partition['val'], **params)

    model = PPGModel_plus()
    # model.load_weights('', by_name=True, skip_mismatch=False)
    model.summary()

    model.compile(optimizer=Adam(1e-4), loss='mse',metrics=['mse'])
    model_checkpoint_1 = ModelCheckpoint(
        paths['model_dir'] + os.sep + paths['train_date'] + os.sep + paths['model_name'] + '_Weights_{epoch:02d}_{val_loss:.8f}.hdf5',
        monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    tb_cb = TensorBoard(paths['log_dir'] + os.sep + paths['train_date'])
    early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
    lr_reducer = ReduceLROnPlateau(factor=0.1, cooldown=5, patience=20, min_lr=1e-8)
    calls_backs = [model_checkpoint_1, tb_cb, early_stop, lr_reducer]

    model.fit_generator(generator=train_generator,
                        validation_data=val_generator,
                        callbacks=calls_backs,
                        epochs=200,
                        use_multiprocessing=True,
                        # workers=4,
                        # max_queue_size=8,
                        shuffle=True)
    model.save_weights(paths['model_dir'] + os.sep + paths['train_date'] + os.sep + paths['model_name'] +'.hdf5')


if __name__ == "__main__":
    main()





