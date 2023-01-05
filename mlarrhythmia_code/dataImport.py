import pandas as pd
import matplotlib.pyplot as plt
from SeabornTheme import *

def GetData():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data'
    df = pd.read_csv(url, header=None, na_values='?')
    df.columns = [line.rstrip() for line in open('mlarrhythmia_dataaddons/arrhythmia.headers')]
    classes = [line.rstrip() for line in open('mlarrhythmia_dataaddons/arrhythmia.classes')]
    return classes, df

def GetDataClean():
    classes, df = GetData()
    df = df.drop('J', axis=1)
    df = df.fillna(df.median())
    return classes, df

if __name__ == "__main__":
    _, df =  GetData()
    mask = df.isnull()
    missing = mask.sum()
    percent_missing = missing / len(df)
    percent_missing_nonzero = percent_missing[percent_missing > 0]
    print(percent_missing_nonzero)

    setTheme()
    class_counts = df["Class"].value_counts()
    sns.barplot(x = class_counts.index, y = class_counts.values)
    plt.xlabel('Class')
    plt.ylabel('Number of instances')
    plt.savefig('mlarrhythmia_present/class_distribution.png', format='png')