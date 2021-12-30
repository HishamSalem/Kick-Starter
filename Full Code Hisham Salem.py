# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 16:04:10 2021

@author: hisha
"""

# =============================================================================
# ##############################################################################
# #Classification FULL CODE
# ##############################################################################
# ##############################################################################
# #Importing all packages
# ##############################################################################
# =============================================================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso



from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier


from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import IsolationForest
from numpy import where
##############################################################################
#Data Preprocessing
##############################################################################

#Importing the data
df=pd.read_excel('D:\\Downloads\\Kickstarter.xlsx')
#Drop observations
df.drop( df[ (df['state'] !='failed') & (df['state']!='successful')].index , inplace=True)

#conert goal to used
df['goal_in_usd'] =df['goal']*df['static_usd_rate']
#drop unneccessary columns

df=df.drop(columns=['project_id','name','disable_communication','currency','pledged','usd_pledged','goal','static_usd_rate',
                    'deadline','created_at','state_changed_at','launched_at',
                    'state_changed_at_month', 'state_changed_at_day','state_changed_at_yr','state_changed_at_hr',
                    'backers_count','spotlight','staff_pick',
                    'state_changed_at_weekday','launch_to_state_change_days','launched_at_weekday'
           ])
#create 1 column for state
df.loc[df.state =='successful', 'state'] = 0
df.loc[df.state =='failed', 'state'] = 1
#drop na
df=df.dropna()
#specify x and y
y=df['state']
y=y.astype('int')


X=df.drop(columns=['state'])
X=pd.get_dummies(X,columns=['country','category','deadline_weekday','created_at_weekday'])

#Removing anomalies:

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

num_col = X.select_dtypes(include=numerics)
iforest=IsolationForest(n_estimators=100,contamination=0.02)

pred=iforest.fit_predict(num_col)
score=iforest.decision_function(num_col)


non_anom_index=where(pred==1)
X=X.iloc[non_anom_index]
y=y.iloc[non_anom_index]
##############################################################################
#Feature selection:
##############################################################################

lr = LogisticRegression(max_iter=5000)
rfe = RFE(lr, n_features_to_select=50)
model = rfe.fit(X, y)
model.ranking_
rating=pd.DataFrame(list(zip(X.columns,model.ranking_)), columns = ['predictor','ranking'])

###############################################################################
#Classification#
##############################################################################

##############################################################################
#KNN Classifier
##############################################################################
scaler=StandardScaler()
X_std=scaler.fit_transform(X)

#with loops approach
max=0
wanted=100
for i in range (2,10):
    knn = KNeighborsClassifier(n_neighbors=i)
    model2 = knn.fit(X_std,y)
    scores=cross_val_score(estimator=model2,X=X_std,y=y,cv=5)
    cross_score=np.average(scores)
    if cross_score>max:
        max=cross_score
        wanted=i
print(max)
print(wanted)


#with gridsearch appraoch
knn = KNeighborsClassifier()
parammeters = dict(n_neighbors=list(range(1,22,1)))
clf = GridSearchCV(knn, parammeters, cv=5, scoring='accuracy', return_train_score=False,verbose=1,n_jobs=-1)
print('Current DateTime:', datetime.now())
modelKnn=clf.fit(X,y)
print('Current DateTime:', datetime.now())
#19 n neighbors optimal soloution

print(clf.score(X,y))
print(clf.best_params_)


#cv score: 0.7139062387823965
#20 neighbors
##############################################################################
#ANN Classifier
##############################################################################
max=0
wanted=100
for i in range (2,15):
    model3 = MLPClassifier(hidden_layer_sizes=(i),max_iter=1000)
    scores = cross_val_score(estimator=model3, X=X, y=y, cv=5)
    cross_score=np.average(scores)
    if cross_score>max:
        max=cross_score
        wanted=i
print(max)
print(wanted)


ANN=MLPClassifier(max_iter=1000)
parameters = {'hidden_layer_sizes': list(range(2,25,1))
             }

clf = GridSearchCV(ANN, parameters, cv = 5,n_jobs=-1)
print('Current DateTime:', datetime.now())
model13=clf.fit(X,y)
print('Current DateTime:', datetime.now())
print(clf.score(X,y))
print(clf.best_params_)


#CV score is : 0.6989733649221049
# 18 hidden layers
##############################################################################
#Random Forest Classifier
#RandomForest#
##############################################################################
# =============================================================================
# #Before Binary Search Technique
# parameters = {
#     "min_samples_split": list(range(1,12,1)), #1
#     "min_samples_leaf": list(range(1,10,1)), #1
#     "max_depth":list(range(1,26,1)),#4
#     "max_features":['auto','sqrt','log2'],
#     "n_estimators":list(range(1,150,25)) 
#     }
# #list(range(1,32,8))
# RF=RandomForestClassifier()
# 
# =============================================================================

#After Binary Search Technique
parameters = {
    "min_samples_split": list(range(11,12,1)), #11
    "min_samples_leaf": list(range(1,2,1)), #1
    "max_depth":list(range(25,26,1)),#25
    "max_features":['auto'],
    "criterion": ['gini', 'entropy'],
    "n_estimators":list(range(106,110,2)) 
    }

RF=RandomForestClassifier()
clf= GridSearchCV(RF,parameters,cv=5,n_jobs=-1)
print('Current DateTime:', datetime.now())
model1=clf.fit(X,y)
print('Current DateTime:', datetime.now())
print(clf.score(X,y))
print(clf.best_params_)


#Best RF Model:
RF=RandomForestClassifier(random_state=5,n_estimators=108,min_samples_leaf=1, min_samples_split=11,max_depth=25,max_features='auto',criterion='entropy')
model1=RF.fit(X,y)
scores=cross_val_score(estimator=model1,X=X,y=y,cv=5)
cross_score=np.average(scores)
print(cross_score)
#0.7473620474777066

#specify x and y
y_grading=kickstarter_grading_df['state']
y_grading=y_grading.astype('int')

    
X_grading=kickstarter_grading_df.drop(columns=['state'])
X_grading=pd.get_dummies(X_grading,columns=['country','category','deadline_weekday','created_at_weekday'])


y_grading_pred = model1.predict(X_grading)

AS=accuracy_score(y_grading, y_grading_pred)
print("Score for Test set is : "+str(cross_score))
print('Score For Grading is is : '+ str(AS))

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=5)
RF=RandomForestClassifier(random_state=5,n_estimators=108,min_samples_leaf=1, min_samples_split=11,max_depth=25,max_features='auto')
model1_train=RF.fit(X_train,y_train)
y_test_pred=model1_train.predict(X_test)
y_train_pred=model1_train.predict(X_train)

print("CV for test is : "+str(accuracy_score(y_test,y_test_pred)))
print("CV for training is : "+str(accuracy_score(y_train,y_train_pred)))
a=metrics.confusion_matrix(y_test,y_test_pred)
b=metrics.precision_score(y_test,y_test_pred)
c=metrics.recall_score(y_test,y_test_pred)
d=metrics.f1_score(y_test,y_test_pred)
print("Confusion matrix: "+str(a))
print("Precision score: "+str(b))
print("Recall score: "+str(c))
print("F1 score: "+str(d))

#Score for Test set is : 0.7561212319612941
#Score For Grading is is : 0.9478260869565217
#CV for test is : 0.7647762622636994
#CV for training is : 0.9542564102564103
#Confusion matrix: [[ 646  701]
#                   [ 282 2550]]
#Precision score: 0.7843740387573055
#Recall score:    0.9004237288135594
#F1 score:        0.838402104224889



##############################################################################
#Gradient boosting classifier
##############################################################################
# =============================================================================
# =============================================================================
# parameters = {
#     "min_samples_split": list(range(2,20,1)),
#     "min_samples_leaf": list(range(1,10,1)),
#     "max_depth":list(range(2,30,4)),
#     "max_features":['auto','sqrt','log2'],
#     "learning_rate": [0.01,0.05, 0.1,0.15, 0.2,0.25,0.3],
#     "n_estimators":list(range(40,250,10))
#     }
# =============================================================================
# =============================================================================

#After binary Search
parameters = {
    "min_samples_split": list(range(18,19,1)),
    "min_samples_leaf": list(range(11,13,1)),
    "max_depth":list(range(6,7,1)),
    "max_features":['sqrt'],
    "learning_rate": [0.15],
    "n_estimators":list(range(117,119,1))
    }

GB=GradientBoostingClassifier()
clf= GridSearchCV(GB,parameters,cv=5,n_jobs=-1)

print('Current DateTime:', datetime.now())
model2=clf.fit(X,y)
print('Current DateTime:', datetime.now())
print(clf.score(X,y))
print(clf.best_params_)

#final Gradient boosting model
GB=GradientBoostingClassifier(random_state=5,n_estimators=117,min_samples_leaf=11, min_samples_split=18,max_depth=6,max_features='sqrt',learning_rate=0.15)
model1=GB.fit(X,y)
scores=cross_val_score(estimator=model1,X=X,y=y,cv=5)
cross_score=np.average(scores)
print(cross_score)
#CV:0.7564796281999894

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=5)
GB=GradientBoostingClassifier(random_state=5,n_estimators=117,min_samples_leaf=11, min_samples_split=18,max_depth=6,max_features='sqrt',learning_rate=0.15)
model_train=GB.fit(X_train,y_train)
y_test_pred=model_train.predict(X_test)
y_train_pred=model_train.predict(X_train)

print("CV for test is : "+str(accuracy_score(y_test,y_test_pred)))
print("CV for training is : "+str(accuracy_score(y_train,y_train_pred)))
a=metrics.confusion_matrix(y_test,y_test_pred)
b=metrics.precision_score(y_test,y_test_pred)
c=metrics.recall_score(y_test,y_test_pred)
d=metrics.f1_score(y_test,y_test_pred)
print("Confusion matrix: "+str(a))
print("Precision score: "+str(b))
print("Recall score: "+str(c))
print("F1 score: "+str(d))
#CV for test is : 0.7547260110074181
#CV for training is : 0.8138461538461539
#Confusion matrix: [[ 744  638]
#                   [ 361 2436]]
#Precision score: 0.7924528301886793
#Recall score: 0.8709331426528424
#F1 score: 0.8298415942769545