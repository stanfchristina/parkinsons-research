#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors
from sklearn.svm import SVC

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


# In[2]:


main = pd.read_csv("accuracy_tapping_reduced.csv")
main.drop(['Unnamed: 0'], axis=1, inplace=True)
main


# In[3]:


data = main.iloc[:, 3:11]
data = data.values


# In[4]:


labels = main.iloc[:,2]
labels = labels.to_frame()


# In[5]:


encoder = LabelEncoder()
encoder.fit(labels)
# 0 -> "Immediately before Parkinson medication"
# 1 -> "Just after Parkinson medication (at your best)"
encoded_labels = encoder.transform(labels)


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(data, encoded_labels, test_size=0.2)
X_train_norm = (X_train - X_train.mean()) / X_train.std()
X_test_norm = (X_test - X_test.mean()) / X_test.std()


# ## Logistic Regression

# In[7]:


fold = KFold(n_splits=10, shuffle=True, random_state=0)
grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'tol': [1e-4, 1e-3, 1e-2], 'solver': ['newton-cg', 'sag', 'lbfgs']}
lr = GridSearchCV(LogisticRegression(penalty='l2', max_iter=5000), param_grid=grid, cv=fold)
lr.fit(X_train_norm, y_train)


# In[8]:


print("Best parameters set found on development set:")
print(lr.best_params_)
print()
print("Best score found on development set:")
print(lr.best_score_ )
print()

y_true, y_pred = y_test, lr.predict(X_test_norm)
lr_class_report = classification_report(y_true, y_pred)

print(lr_class_report)


# ## Decision Tree

# In[9]:


tuned_parameters = [{"criterion": ["gini", "entropy"], "min_samples_split": [10, 20, 40], "max_depth": [10, 14, 18], "min_samples_leaf": [10, 20, 40]}]
fold = KFold(n_splits=10, shuffle=True, random_state=0)
dt = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=fold)
dt.fit(X_train_norm, y_train)


# In[10]:


print("Best parameters set found on development set:")
print(dt.best_params_)
print()
print("Best score found on development set:")
print(dt.best_score_ )
print()

y_true, y_pred = y_test, dt.predict(X_test_norm)
dt_class_report = classification_report(y_true, y_pred)

print(dt_class_report)


# ## K Nearest Neighbors

# In[11]:


metrics = ['minkowski','euclidean','manhattan'] 
weights = ['uniform','distance'] 
numNeighbors = [5, 7, 10, 15, 20]
param_grid = dict(metric=metrics,weights=weights,n_neighbors=numNeighbors)
fold = KFold(n_splits=10, shuffle=True, random_state=0)
knn = GridSearchCV(neighbors.KNeighborsClassifier(),param_grid=param_grid,cv=fold)
knn.fit(X_train_norm, y_train)


# In[12]:


print("Best parameters set found on development set:")
print(knn.best_params_)
print()
print("Best score found on development set:")
print(knn.best_score_ )
print()

y_true, y_pred = y_test, knn.predict(X_test_norm)
knn_class_report = classification_report(y_true, y_pred)

print(knn_class_report)


# ## Support Vector Classification

# In[13]:


fold = KFold(n_splits=10, shuffle=True, random_state=0)
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
svc = GridSearchCV(SVC(probability=True), param_grid=tuned_parameters, cv=fold)
svc.fit(X_train_norm, y_train)


# In[14]:


print("Best parameters set found on development set:")
print(svc.best_params_)
print()
print("Best score found on development set:")
print(svc.best_score_ )
print()

y_true, y_pred = y_test, svc.predict(X_test_norm)
svc_class_report = classification_report(y_true, y_pred)

print(svc_class_report)


# ## Artificial Neural Network

# In[15]:


from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

ann = Sequential()
ann.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=8)) # First hidden layer
ann.add(Dense(4, activation='relu', kernel_initializer='random_normal')) # Second hidden layer
ann.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal')) # Output layer

# Optimize neural network with Adam (Adaptive moment estimation), combination of RMSProp and Momentum.
# Momentum takes past gradients into account in order to smooth out the gradient descent.
ann.compile(optimizer='adam',loss='binary_crossentropy', metrics =['accuracy'])

ann.fit(X_train_norm, y_train, batch_size=10, epochs=100, verbose=0)


# In[16]:


ann_eval = ann.evaluate(X_train_norm, y_train, verbose=0)
y_pred = ann.predict_classes(X_test_norm, batch_size=10, verbose=0).flatten()
ann_class_report = classification_report(y_test.astype(bool), y_pred.astype(bool))

print('Loss and accuracy:')
print(ann_eval)
print()
print(ann_class_report)


# ## ROC Plot

# In[18]:


from sklearn import metrics

plt.figure()

models = [{'label': 'LR', 'model': lr},
          {'label': 'DT', 'model': dt},
          {'label': 'KNN', 'model': knn},
          {'label': 'SVC', 'model': svc},
          {'label': 'ANN', 'model': ann}]

for m in models:
    model = m['model']
    y_pred = model.predict(X_test_norm)
    
    if model is ann:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test_norm))
    else:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test_norm)[:,1])
        
    auc = metrics.roc_auc_score(y_test,model.predict(X_test_norm))
    plt.plot(fpr, tpr, label='%s (area = %0.2f)' % (m['label'], auc))

plt.plot([0, 1], [0, 1],'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Precision (False Positive Rate)')
plt.ylabel('Recall (True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

