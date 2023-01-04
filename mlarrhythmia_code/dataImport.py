import pandas as pd
import numpy as np

def GetDataFrame():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data'
    df = pd.read_csv(url, header=None, na_values='?')
    df.columns = [line.rstrip() for line in open('mlarrhytmia_dataaddons/arrhythmia.headers')]

    classes = [line.rstrip() for line in open('mlarrhytmia_dataaddons/arrhythmia.classes')]
    for i, c in enumerate(classes, start=1):
        df["Class"] = df["Class"].replace(i, c)

    return df