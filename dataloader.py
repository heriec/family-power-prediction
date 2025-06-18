import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import torch


def create_sliding_window_data(data, n_in=1, n_out=1):
    X, y = [], []
    for i in range(len(data) - n_in - n_out + 1):
        X.append(data[i: i + n_in])
        y.append(data[i + n_in: i + n_in + n_out, 0])
    return np.array(X), np.array(y)


def load_data(file_path='data/train.csv', n_in=1, n_out=1):
    df = pd.read_csv(file_path,  parse_dates=[
                     'DateTime'], index_col='DateTime', na_values=['?', ''])
    df = df.fillna(df.mean())

    df_resample = df.resample('D').mean()
    values = df_resample.values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    print(f"Scaled data shape: {scaled.shape}")

    train_X, train_y = create_sliding_window_data(scaled, n_in, n_out)
    train_X = train_X.reshape(train_X.shape[0], -1)
    print(f"data shape: {train_X.shape}, labels shape: {train_y.shape}")

    return train_X, train_y


def create_dataloaders(X, y, batch_size):

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
