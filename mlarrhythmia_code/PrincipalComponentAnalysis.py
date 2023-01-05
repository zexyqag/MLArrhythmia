from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from DataImport import *
from SeabornTheme import *

def GetDataPCA(rv = 0):
    #Get the data and fix missing values
    classes, df = GetDataClean()
    data = df.drop("Class", axis=1)

    #Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    #PCA the data
    if (rv == 0):
        pca = PCA()
    else:
        pca = PCA(rv)
    pca.fit(data_scaled)

    data_transformed = pca.transform(data_scaled)
    df_transformed = pd.concat([pd.DataFrame(data_transformed), pd.DataFrame(df["Class"], columns=['Class'])],axis=1) 

    return classes, pca, df_transformed

if __name__ == "__main__":
    print(len(GetDataPCA(0.95)[1].explained_variance_ratio_))
    print(len(GetDataPCA(0.99)[1].explained_variance_ratio_))

    _, pca, _ = GetDataPCA()

    #Plot the data
    explained_variance = pca.explained_variance_ratio_
    cumulative_explained_variance = [explained_variance[:i+1].sum() for i in range(len(explained_variance))]
    setTheme()
    # Create a line plot of the cumulative explained variance ratio
    sns.lineplot(x = [i for i in range(len(cumulative_explained_variance))], y = cumulative_explained_variance)
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance ratio')

    plt.savefig('mlarrhythmia_present/variance_plot.png', format='png')