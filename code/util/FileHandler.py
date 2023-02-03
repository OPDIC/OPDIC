# -*- coding: utf-8 -*-


import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class FileHandler:
    def __init__(self):
        self.PATH = ''

    def readCompData(self, input, label_idx=None, header=None, index_col=None):
        data = np.array(pd.read_csv(os.path.join(self.PATH, input), header=header, index_col=index_col))
        scaler = MinMaxScaler()
        if label_idx != -1:
            label = data[:, label_idx]
            data = scaler.fit_transform(data)
            data[:, label_idx] = label
        else:
            data = scaler.fit_transform(data)
        return pd.DataFrame(data)
