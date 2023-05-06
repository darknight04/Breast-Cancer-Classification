from sklearn.datasets import load_breast_cancer

import pandas as pd

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)

print(df.describe())

       mean radius  mean texture  ...  worst symmetry  worst fractal dimension
count   569.000000    569.000000  ...      569.000000               569.000000
mean     14.127292     19.289649  ...        0.290076                 0.083946
std       3.524049      4.301036  ...        0.061867                 0.018061
min       6.981000      9.710000  ...        0.156500                 0.055040
25%      11.700000     16.170000  ...        0.250400                 0.071460
50%      13.370000     18.840000  ...        0.282200                 0.080040
75%      15.780000     21.800000  ...        0.317900                 0.092080
max      28.110000     39.280000  ...        0.663800                 0.207500

[8 rows x 30 columns]
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

X = data.data

y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf = DecisionTreeClassifier(random_state=0)

clf.fit(X_train, y_train)

DecisionTreeClassifier(random_state=0)
y_pred = clf.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))

Classification Report:
               precision    recall  f1-score   support

           0       0.84      0.94      0.89        63
           1       0.96      0.90      0.93       108

    accuracy                           0.91       171
   macro avg       0.90      0.92      0.91       171
weighted avg       0.92      0.91      0.91       171

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

Confusion Matrix:
 [[59  4]
 [11 97]]
#PART 3
 
#PCA
 
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=0)

clf = DecisionTreeClassifier(random_state=0)

clf.fit(X_train, y_train)

DecisionTreeClassifier(random_state=0)
y_pred = clf.predict(X_test)

print("Classification Report after PCA:\n", classification_report(y_test, y_pred))

Classification Report after PCA:
               precision    recall  f1-score   support

           0       0.88      0.94      0.91        63
           1       0.96      0.93      0.94       108

    accuracy                           0.93       171
   macro avg       0.92      0.93      0.93       171
weighted avg       0.93      0.93      0.93       171

print("Confusion Matrix after PCA:\n", confusion_matrix(y_test, y_pred))

Confusion Matrix after PCA:
 [[ 59   4]
 [  8 100]]
#FEATURE SELECTION
 
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=10)

X_new = selector.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=0)

clf = DecisionTreeClassifier(random_state=0)

clf.fit(X_train, y_train)

DecisionTreeClassifier(random_state=0)
y_pred = clf.predict(X_test)

print("Classification Report after Feature Selection:\n", classification_report(y_test, y_pred))

Classification Report after Feature Selection:
               precision    recall  f1-score   support

           0       0.91      0.92      0.91        63
           1       0.95      0.94      0.95       108

    accuracy                           0.94       171
   macro avg       0.93      0.93      0.93       171
weighted avg       0.94      0.94      0.94       171

print("Confusion Matrix after Feature Selection:\n", confusion_matrix(y_test, y_pred))

Confusion Matrix after Feature Selection:
 [[ 58   5]
 [  6 102]]
#OVERSAMPLING
 
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=0)

X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)

clf = DecisionTreeClassifier(random_state=0)

clf.fit(X_train, y_train)

DecisionTreeClassifier(random_state=0)
y_pred = clf.predict(X_test)

print("Classification Report after Oversampling:\n", classification_report(y_test, y_pred))
Classification Report after Oversampling:
               precision    recall  f1-score   support

           0       0.93      0.98      0.95        96
           1       0.98      0.94      0.96       119

    accuracy                           0.96       215
   macro avg       0.96      0.96      0.96       215
weighted avg       0.96      0.96      0.96       215


