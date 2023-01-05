---

marp: true
theme: gaia
class: invert
paginate: false
style: |
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }

---

# Detection / classification of arrhythmia
<span style="color:lightgrey">By:</span> Martin Dahl Sørensen

---
<style scoped>section { font-size: 20px; }.columns{font-size: 15px;} </style>

|  |  |  |  |  |  |
|---|---|---|---|---|---|
| Data Set Characteristics:   | Multivariate | Number of Instances: | 452 | Area: | Life |
| Attribute Characteristics: | Categorical, Integer, Real | Number of Attributes: | 279 | Date Donated | 1998-01-01 |
| Associated Tasks: | Classification | Missing Values? | Yes | Number of Web Hits: | 418004 |

<div class ="columns">
<div>

![height:400px](class_distribution.png)

</div>
<div>
&nbsp;

1. Normal
2. Ischemic changes (Coronary Artery Disease)
3. Old Anterior Myocardial Infarction
4. Old Inferior Myocardial Infarction
5. Sinus tachycardy
6. Sinus bradycardy
7. Ventricular Premature Contraction (PVC)
8. Supraventricular Premature Contraction
9. Left bundle branch block
10. Right bundle branch block
11. 1\. degree AtrioVentricular block	
12. 2\. degree AV block
13. 3\. degree AV block
14. Left ventricule hypertrophy
15. Atrial Fibrillation or Flutter
16. Others

</div>
</div>

<!-- 
The data set I've worked with is the arrhythmia Data Set.

The goal of this dataset, as described on the Machine Learning Repository, is to distinguish between the presence and absence of cardiac arrhythmia and classify it in one of the 16 groups.

There are 279 attributes with 452 instances where some may contain missing attributes. these missing values are denoted with a question mark
-->

---
#  Importing the data 
```python
def GetData():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data'
    df = pd.read_csv(url, header=None, na_values='?')
    df.columns = [line.rstrip() for line in open('mlarrhythmia_dataaddons/arrhythmia.headers')]
    classes = [line.rstrip() for line in open('mlarrhythmia_dataaddons/arrhythmia.classes')]
    return classes, df
```
```
              Age         Sex      Height      Weight  QRS duration  P-R interval  ...  V6 S' wave   V6 P wave   V6 T wave     V6 QRSA    V6 QRSTA       Class
count  452.000000  452.000000  452.000000  452.000000    452.000000    452.000000  ...       452.0  452.000000  452.000000  452.000000  452.000000  452.000000
mean    46.471239    0.550885  166.188053   68.170354     88.920354    155.152655  ...         0.0    0.514823    1.222345   19.326106   29.473230    3.880531
std     16.466631    0.497955   37.170340   16.590803     15.364394     44.842283  ...         0.0    0.347531    1.426052   13.503922   18.493927    4.407097
min      0.000000    0.000000  105.000000    6.000000     55.000000      0.000000  ...         0.0   -0.800000   -6.000000  -44.200000  -38.600000    1.000000
25%     36.000000    0.000000  160.000000   59.000000     80.000000    142.000000  ...         0.0    0.400000    0.500000   11.450000   17.550000    1.000000
50%     47.000000    1.000000  164.000000   68.000000     86.000000    157.000000  ...         0.0    0.500000    1.350000   18.100000   27.900000    1.000000
75%     58.000000    1.000000  170.000000   79.000000     94.000000    175.000000  ...         0.0    0.700000    2.100000   25.825000   41.125000    6.000000
max     83.000000    1.000000  780.000000  176.000000    188.000000    524.000000  ...         0.0    2.400000    6.000000   88.800000  115.900000   16.000000
```
<!--
Importing the data is easy with pandas. 
For added convenience I have local list of attributes names I add to the DataFrame as headers.
And I supply a list of the class names as an array in the get method.

Then using the describe method on the DataFrame it shows me a distribution of the first five and the last four attributes, as the last column is the labels.

There are a lot of attributes and I have no idea what any of them mean except for the first four.
I could try and plot some of the attributes to see if there any clear distinction between classes, but since there are so many of them I don't think it makes sense to try and manually figure out which attributes have the most influence. 
-->
---
# Cleaning the data


```python
_, df =  GetData()
mask = df.isnull()
missing = mask.sum()
percent_missing = missing / len(df)
percent_missing_nonzero = percent_missing[percent_missing > 0]
print(percent_missing_nonzero)
```

<div class ="columns">
<div>

```
T             0.017699
P             0.048673
QRST          0.002212
J             0.831858
Heart rate    0.002212
```
</div>
<div>

```py
def GetDataClean():
    classes, df = GetData()
    df = df.drop('J', axis=1)
    return classes, df
```
</div>
</div>



<!--
As you can see 5 of the attributes contain missing data with J being the worst as 83% of its data is missing. for this reason im gonna drop this column from the data set


-->

---
# Principal Component Analysis (PCA)

<div class ="columns">
<div>

95% - 103 axes
99% - 154 axes
![height:425px](variance_plot.png)


</div>
<div>

```python
def GetDataPCA(rv = 0):
    #Get the data and fix missing values
    classes, df = GetDataClean()
    df = df.fillna(df.median())
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
    df_transformed = pd.concat([pd.DataFrame(data_transformed), 
    pd.DataFrame(df["Class"], columns=['Class'])],axis=1) 

    return classes, pca, df_transformed
```
```python
print(len(GetDataPCA(0.95)[0].explained_variance_ratio_))
print(len(GetDataPCA(0.99)[0].explained_variance_ratio_))
```

</div>
</div>

<!--
A way to reduce dimensionality of the data is with Principal Component Analysis. The main goal of PCA is to reframe the data to make it easier to separate things out and cluster things, and as a result the resulting axes are also ordered from the most to least useful, and with this we can then reduce the dimensionality by discarding the maybe not so important axes.

Before applying PCA it is a must to standardized the data, so all of the attributes are centered around zero and have a standard deviation of one. This is because PCA is sensitive to the scale of the features, and features on larger scales can dominate the result.

By taking the cumulative explained variance and plotting we can se that the first 103 axes explain 95% of the data with 154 explaining 99%
-->
---

## intra/inter Class distances

```python
df = GetArrhythmiaDataFrame()

# Extract the labels and features from the DataFrame
labels = df['Class'].values
features = df.drop('Class', axis=1).values

# Calculate the intra-class distances for each class
intra_class_dists = {}
for label in np.unique(labels):
    class_mask = (labels == label)
    class_features = features[class_mask]
    intra_class_dists[label] = np.mean(pdist(class_features))

# Print the average intra-class distances
print(intra_class_dists)

inter_class_dists = []
for label_1 in np.unique(labels):
    for label_2 in np.unique(labels):
        if label_1 == label_2:
            continue
        class_1_mask = (labels == label_1)
        class_2_mask = (labels == label_2)
        class_1_features = features[class_1_mask]
        class_2_features = features[class_2_mask]
        combined_features = np.concatenate((class_1_features, class_2_features))
        inter_class_dists.append(np.min(pdist(combined_features)))

# Print the minimum inter-class distance
print(np.min(inter_class_dists))
```
---

<div class ="columns">
<div>

## Intra-Class distance
<div class ="columns">
<div>

1. 227
2. 323
3. 332
4. 229
5. 367
6. 233
7. 294
8. 362

</div>
<div>

9. 482
10. 282
11. n/a
12. n/a
13. n/a
14. 305
15. 266
16. 304

</div>
</div>

</div>
<div>

## Inter-class distance
- count 156
- mean 117
- median 113
- std 42
- min 74
- max 262

</div>
</div>

---

## Quadratic

```python
#Get the DataFrame
df = GetArrhythmiaDataFrame()

data = df.drop("Class", axis=1)
lables = df.loc[:,"Class"]

#Normalize data to alleviate features scale affecting the results
data_normalized = scale(data)

X_train, X_test, y_train, y_test = train_test_split(data_normalized, lables, test_size=0.2)

#Apply PCA
pca = PCA(0.95)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)
print(pca.n_components_)

clf = SVC(kernel='poly', degree=2)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)
```
---

95% of retained variance
Components: 93
accuracy: 63%