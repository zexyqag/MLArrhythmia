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

## Importing the data 
```python
df = pd.read_csv('arrhythmia.data', header=None, na_values='?')
df.columns = [line.rstrip() for line in open('arrhythmia.headers')]
df = df.fillna(df.median())
print(df.describe)
```
Drop missing rows/columns vs impute data (mean/median/other) vs annotate

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
---

![](DistributionOffClasses.png)


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

## Principal Component Analysis

```python
df = GetArrhythmiaDataFrame()

scaler = StandardScaler()
df_normalized = scaler.fit_transform(df)

pca = PCA()
pca.fit(df_normalized)

explainedVariance = pca.explained_variance_ratio_

# Get the indices of the sorted feature weights in the first PC
sorted_indices = np.argsort(pca.components_[0])[::-1]

# Get the label of the feature with the highest weight in the first PC
highest_weight_feature_label = df.columns[sorted_indices[0:25]]

print(highest_weight_feature_label)

setTheme()
sns.lineplot(data = explainedVariance, markers=True)
plt.savefig('pca.png', format='png')
```

---

## Principal Component Analysis

<div class ="columns">
<div>

![](pca.png)
</div>
<div>
<div class ="columns">
<div>

1. AVL QRSA
2. AVL R wave
3. AVL R wave
4. AVR QRSTA
5. AVF S wave
6. DIII S wave
7. AVL QRSTA
8. AVL Number of intrinsic deflections
10. DII S wave
</div>
<div>

11. AVR T wave
12. DIII JJ wave
13. AVL S wave
14. V4 S wave
15. DI QRSA
16. V1 Q wave
17. V2 Q wave
18. Age
19. V5 S wave
20. V3 Q wave
</div>
</div>
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