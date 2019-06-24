import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(filename):
    df = pd.read_csv(filename, sep="\t", names=['1', '2', 'target'])
    print(df)
    data = []
    target = []
    for i in range(len(df)):
        d = df.iloc[i]
        ds = [d['1'], d['2']]
        t = int(d['target'])
        if t == 0:
            t = -1
        data.append(ds)
        target.append(t)
    return np.mat(data), np.mat(target).tolist()[0]


if __name__ == '__main__':
    dataMatrix, target = load_data('data1.txt')
    print(dataMatrix)
    print(target)
