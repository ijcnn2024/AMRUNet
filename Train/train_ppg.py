"""
ppg+ecg data samplerate 125, segment_data len 1024 for training
"""

import os
import datetime
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from Train.DataGenerator_ppg import DataGenerator
from Net.models import AMRUNetBP


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
    params = {'batch_size':32,
              'data_len': 1024,
              'n_channels': 1,
              'data_path': '',
              'shuffle': True,
              'ds': True,
              'use_filter': True,
              'error_log': 'PPGErrorLog.csv'}

    paths = {'model_name': 'AMRUNetBP',
             'model_dir': '/PPGtoABP/Model/AMRUNetBP',
             'log_dir': '/PPGtoABP/Logs/AMRUNetBP',
             'train_csv': '/PPGtoABP/csv/select_data/train.csv',
             'val_csv': '/PPGtoABP/csv/select_data/val.csv',
             'model_weight': 'AMRUNetBP.hdf5',
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
    val_generator = DataGenerator(partition['val'], **params)

    model = AMRUNetBP(length=1024, n_channel=1)

    model.load_weights('/PPGtoABP/Model/AMRUNetBP.hdf5', by_name=True, skip_mismatch=False)

    model.summary()
    # model.compile(optimizer=Adam(1e-4), loss='mse',metrics=['mse'])
    model.compile(optimizer=Adam(1e-4), loss='mean_absolute_error', metrics=['mean_squared_error'], loss_weights=[1., 0.9, 0.8, 0.7, 0.6])
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
                        epochs=2000,
                        use_multiprocessing=True,
                        # workers=4,
                        # max_queue_size=8,
                        shuffle=True)
    model.save_weights(paths['model_dir'] + os.sep + paths['train_date'] + os.sep + paths['model_name'] +'.hdf5')


if __name__ == "__main__":
    main()





