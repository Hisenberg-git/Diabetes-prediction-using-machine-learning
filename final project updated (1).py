
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


diabetes1 = pd.read_csv("diabetes.csv")
print(diabetes1.columns)


# In[3]:


diabetes1.head()


# print("dimension of diabetes data: {}".format(diabetes1.shape))

# print(diabetes1.groupby('Outcome').size())

# import seaborn as sns
# 
# sns.countplot(diabetes1['Outcome'],label="Count")

# In[7]:


diabetes1.info()


# In[8]: Splitting the dataset for training and testing


from sklearn.model_selection import train_test_split 

feature_col_names = ['Pregnancies', 'Glucose' , 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
predicted_class_names = ['Outcome']



X_train, X_test, y_train, y_test = train_test_split(diabetes1.loc[:, diabetes1.columns != 'Outcome'], diabetes1['Outcome'], stratify=diabetes1['Outcome'], random_state=66)


# In[9]:


print("{0:0.2f}% in training set".format((len(X_train)/len(diabetes1.index))*100))
print("{0:0.2f}% in test set".format((len(X_test)/len(diabetes1.index))*100))


# In[10]:


print ("Origional True: {0} ({1:0.2f}%)".format(len(diabetes1.loc[diabetes1['Outcome'] == 1]),(len(diabetes1.loc[diabetes1['Outcome'] == 1]) / len(diabetes1.index))*100.0))
print ("Origional False: {0} ({1:0.2f}%)".format(len(diabetes1.loc[diabetes1['Outcome'] == 0]),(len(diabetes1.loc[diabetes1['Outcome'] == 0]) / len(diabetes1.index))*100.0))
print(" ")
print ("Training True: {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]),(len(y_train[y_train[:] == 1]) / len(y_train))*100.0))
print ("Training False: {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]),(len(y_train[y_train[:] == 0]) / len(y_train))*100.0))
print(" ")
print ("Testing True: {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]),(len(y_test[y_test[:] == 1]) / len(y_test))*100.0))
print ("Testing False: {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]),(len(y_test[y_test[:] == 0]) / len(y_test))*100.0))


# In[11]:K-means algorithm 


from sklearn.cluster import KMeans

training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10


# In[12]:


kmeans = KMeans(n_clusters=2)  
kmeans.fit(X_train) 


# In[13]:


print(kmeans.cluster_centers_)


# In[14]:


#prediction using kmeans and accuracy
kpred = kmeans.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, kpred)*100
print(accuracy)


# In[15]:confusion matrix 


#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, kpred)
print(cm)


# In[16]:


import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV


# In[17]:


diabetes = pd.read_csv("diabetes.csv")
diabetes.describe()


# In[18]:


# Detailed distribution of the features in the dataset
sns.pairplot(data=diabetes, hue='Outcome')
plt.show()


# In[19]:


print("Feature Engineering")
feature_names = diabetes.columns[:8]
feature_names


# In[20]:


X = diabetes[feature_names]
y = diabetes.Outcome


# In[21]:


print("Correlation Matrix: \n'it provides useful insight into relationships between pairs of variables '")
sns.heatmap(
    data=X.corr(),
    annot=True,
    fmt='.2f',
    cmap='RdYlGn'
)

fig = plt.gcf()
fig.set_size_inches(10, 8)

plt.show()


# In[22]:


print("Recursive Feature Elimination with Cross Validation:\nThe goal of Recursive Feature Elimination (RFE) is to select features by feature ranking with recursive feature elimination.\nFor more confidence of features selection I used K-Fold Cross Validation with Stratified k-fold.")


# In[23]:


diabetes_mod = diabetes[(diabetes.BloodPressure != 0) & (diabetes.BMI != 0) & (diabetes.Glucose != 0)]
diabetes_mod.shape


# In[25]:


X_mod = diabetes_mod[feature_names]
y_mod = diabetes_mod.Outcome

strat_k_fold = StratifiedKFold(
    n_splits=10,
    random_state=42
)

logreg_model = LogisticRegression()

rfecv = RFECV(
    estimator=logreg_model,
    step=1,
    cv=strat_k_fold,
    scoring='accuracy'
)
rfecv.fit(X_mod, y_mod)

plt.figure()
plt.title('RFE with Logistic Regression')
plt.xlabel('Number of selected features')
plt.ylabel('10-fold Crossvalidation')

# grid_scores_ returns a list of accuracy scores
# for each of the features selected
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()

print('rfecv.grid_scores_: {grid_scores}'.format(grid_scores=rfecv.grid_scores_))

# support_ is another attribute to find out the features
# which contribute the most to predicting
new_features = list(filter(
    lambda x: x[1],
    zip(feature_names, rfecv.support_)
))

print('rfecv.support_: {support}'.format(support=rfecv.support_))

# Features are the most suitable for predicting the response class
new_features = list(map(operator.itemgetter(0), new_features))
print('\nThe most suitable features for prediction: {new_features}'.format(new_features=new_features))


# In[26]:


print(" Data standardization:\nStandardize features by removing the mean and scaling to unit variance")
# Features chosen based on RFECV result
best_features = [
    'Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction'
]

X = StandardScaler().fit_transform(X[best_features])


# In[27]:


# Split your data into training and testing (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.20
)


# In[28]:


#principal component analysis (main goal is to identify patterns in data. PCA aims to detect the correlation between variables.)
#If a strong correlation between variables exists, the attempt to reduce the dimensionality only makes sense.)
pca = PCA(n_components=2)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print(pca.explained_variance_ratio_)
print('PCA sum: {:.2f}%'.format(sum(pca.explained_variance_ratio_) * 100))


# In[29]:


from sklearn.model_selection import GridSearchCV

c_values = list(np.arange(1, 100))

param_grid = [
    {
        'C': c_values,
        'penalty': ['l1'],
        'solver': ['liblinear'],
        'multi_class': ['ovr'],
        'random_state': [42]
    },
    {
        'C': c_values,
        'penalty': ['l2'],
        'solver': ['liblinear', 'newton-cg', 'lbfgs'],
        'multi_class': ['ovr'],
        'random_state': [42]
    }
]

grid = GridSearchCV(
    LogisticRegression(),
    param_grid,
    cv=strat_k_fold,
    scoring='f1'
)
grid.fit(X, y)

# Best LogisticRegression parameters
print(grid.best_params_)
# Best score for LogisticRegression with best parameters
print('Best score: {:.2f}%'.format(grid.best_score_ * 100))


# In[30]:


#Model evaluation
log_reg = LogisticRegression(
    # Parameters chosen based on GridSearchCV result
    C=1,
    multi_class='ovr',
    penalty='l2',
    solver='newton-cg',
    random_state=42
)
log_reg.fit(X_train, y_train)

log_reg_predict = log_reg.predict(X_test)
log_reg_predict_proba = log_reg.predict_proba(X_test)[:, 1]


# In[31]:


print('Accuracy: {:.2f}%'.format(accuracy_score(y_test, log_reg_predict) * 100))
print('Classification report:\n\n', classification_report(y_test, log_reg_predict))
print('Training set score: {:.2f}%'.format(log_reg.score(X_train, y_train) * 100))
print('Testing set score: {:.2f}%'.format(log_reg.score(X_test, y_test) * 100))
#confusion matrix
print("confusion matrix")
outcome_labels = sorted(diabetes.Outcome.unique())

sns.heatmap(
    confusion_matrix(y_test, log_reg_predict),
    annot=True,
    xticklabels=outcome_labels,
    yticklabels=outcome_labels
)


# In[32]:


print("this is a new cell")


# # RANDOM FOREST

# In[33]:



from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state = 42) # create random forest object
sv=rf_model.fit(X_train, y_train.ravel())

with open('sv.pkl','wb') as m:
        pickle.dump(sv,m)


# # prediction accuracy on training data

# In[34]:


rf_predict_train = rf_model.predict(X_train)
        
with open('sv.pkl','rb') as mod:
        p=pickle.load(mod)
from sklearn import metrics
print("Accuracy : {0:.4f}".format(metrics.accuracy_score(y_train, rf_predict_train)))
print()


# # prediction accuracy on testing data

# In[35]:


rf_predict_test = rf_model.predict(X_test)
print("Accuracy : {0:.4f}".format(metrics.accuracy_score(y_test, rf_predict_test)))
print()


# # NAIVE bayes

# In[36]:


from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train, y_train.ravel())


# # prediction accuracy on training data

# In[37]:


nb_predict_train = nb_model.predict(X_train)
from sklearn import metrics
print("Accuracy : {0:.4f}".format(metrics.accuracy_score(y_train, nb_predict_train)))
print()


# # prediction accuracy on testing data

# In[38]:


nb_predict_text = nb_model.predict(X_test)
from sklearn import metrics
print("Accuracy : {0:.4f}".format(metrics.accuracy_score(y_test, nb_predict_text)))
print()


# In[39]:


from sklearn import model_selection
from sklearn.ensemble import VotingClassifier


estimators = []
model1 = GaussianNB()
estimators.append(('gauss', model1))
model2 = RandomForestClassifier()
estimators.append(('ran', model2))
model3 = LogisticRegression()
estimators.append(('lin', model3))


X = diabetes[feature_names].values
y = diabetes.Outcome.values

ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X, y)
print(results.mean()*100)


# In[41]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train.ravel())


# In[43]:


y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import accuracy_score
svmaccuracy = accuracy_score(y_test, y_pred)*100
print(svmaccuracy)


# In[57]:


from sklearn import model_selection
from sklearn.ensemble import VotingClassifier


estimators = []
model1 = GaussianNB()
estimators.append(('gauss', model1))
model2 = RandomForestClassifier(n_estimators=100, random_state=1)
estimators.append(('ran', model2))
model3 =  LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state=1)
estimators.append(('lin', model3))
model4=SVC(kernel = 'rbf', random_state = 0)
estimators.append(('svc',model4))


X = diabetes[feature_names].values
y = diabetes.Outcome.values

ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X, y)
print(results.mean()*100)


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
 
# data to plot
n_groups = 3
lr = (73.59,83.77,78.53)
rf = (74.03,79.89,77.62)
nb=(74.46,81.17,79.11)
 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.25
bar_width_ = 0.25
opacity = 0.8

 
rects1 = plt.bar(index, lr, bar_width,
alpha=opacity,
color='b',
label='Logistic Regression')
 
rects2 = plt.bar(index + bar_width, rf, bar_width,
alpha=opacity,
color='g',
label='Random Forest')

rects3 = plt.bar(index + bar_width+bar_width_, nb, bar_width,
alpha=opacity,
color='y',
label='Naive Bayes')
 
plt.xlabel('Methods')
plt.ylabel('Accuracy')
plt.title('Accuracy by different methods')
plt.xticks(index + bar_width, ('Without RFE', 'With RFECV', 'With RFE'))
plt.legend()
plt.yticks(np.arange(0, 100, step=5))
 
plt.tight_layout()
plt.show()

plt.show()

