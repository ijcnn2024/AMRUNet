import numpy as np
import pandas as pd


def cal_min_max(data_folder, csv_path):
    df_csv = pd.read_csv(csv_path)
    data_first = np.loadtxt(data_folder + '\\' + df_csv.iloc[0, 0])
    mx_ppg = np.max(data_first[:, 0], axis=0).astype(np.float64)
    mn_ppg = np.min(data_first[:, 0], axis=0).astype(np.float64)
    mx_abp = np.max(data_first[:, 1], axis=0).astype(np.float64)
    mn_abp = np.min(data_first[:, 1], axis=0).astype(np.float64)
    for i in range(len(df_csv)):
        path = data_folder + '\\' + df_csv.iloc[i, 0]
        data = np.loadtxt(path)
        max_ppg = np.max(data[:, 0], axis=0).astype(np.float64)
        min_ppg = np.min(data[:, 0], axis=0).astype(np.float64)
        max_abp = np.max(data[:, 1], axis=0).astype(np.float64)
        min_abp = np.min(data[:, 1], axis=0).astype(np.float64)
        mx_ppg = np.maximum(max_ppg, mx_ppg)
        mn_ppg = np.minimum(min_ppg, mn_ppg)
        mx_abp = np.maximum(max_abp, mx_abp)
        mn_abp = np.minimum(min_abp, mn_abp)
    print(mx_ppg, mn_ppg, mx_abp, mn_abp)
    return mx_ppg, mn_ppg, mx_abp, mn_abp

def cal_mean_std(data_folder, csv_path):
    df_csv = pd.read_csv(csv_path)
    data_ppg, data_abp = [], []
    for i in range(len(df_csv)):
        path = data_folder + '\\' + df_csv.iloc[i, 0]
        data = np.loadtxt(path)
        data_ppg.append(data[:, 0])
        data_abp.append(data[:, 1])
    mean_ppg = np.mean(np.array(data_ppg))
    std_ppg = np.std(np.array(data_ppg))
    mean_abp = np.mean(np.array(data_abp))
    std_abp = np.std(np.array(data_abp))

    print(mean_ppg, std_ppg, mean_abp, std_abp)
    return mean_ppg, std_ppg, mean_abp, std_abp


if __name__ == '__main__':
    data_folder = 'C:\\PPGDatabase\\MIMIC_segment\\train'
    csv_path = '/csv/select_data/bak/test.csv'
    # mx_ppg, mn_ppg, mx_abp, mn_abp = cal_min_max(data_folder, csv_path)
    mean_ppg, std_ppg, mean_abp, std_abp = cal_mean_std(data_folder, csv_path)
    '''
               max_ppg     min_ppg     max_abp       min_abp
    train.csv: 4.0         0.0         196.52894596  50.05538916
    val.csv:   3.98729228  0.00195503  199.34236362  50.11400203
    test.csv:  3.62561095  0.60606061  126.79929469  52.60510030
    
                mean_ppg           std_ppg             mean_abp           std_abp
    train.csv:  1.59019869386979   0.6977130624997815  84.41247576848023  20.99564744196556
    val.csv:    1.61175869630853   0.6854227120044523  84.53310276130706  20.78163310358141
    test.csv:   1.69941779909886   0.5401029996442075  81.37523427840382  14.64811447960598   
    '''