# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 15:22:00 2021

@author: hisha
"""

#packages for classification
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
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


#packages for clustering
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score
from scipy.stats import f
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from numpy import where
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# #Classification
# #lines 56 -156
# =============================================================================



# =============================================================================
# #Data Preprocessing
# =============================================================================


#Importing the data from desired path
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

df.loc[df.state =='successful', 'state'] = 0
df.loc[df.state =='failed', 'state'] = 1
df=df.dropna()

#specify x and y
y=df['state']
y=y.astype('int')

    
X=df.drop(columns=['state'])
X=pd.get_dummies(X,columns=['country','category','deadline_weekday','created_at_weekday'])

#Removing anomalies:

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

num_col = X.select_dtypes(include=numerics)
iforest=IsolationForest(n_estimators=100,contamination=0.02,random_state=5)
pred=iforest.fit_predict(num_col)
score=iforest.decision_function(num_col)
non_anom_index=where(pred==1)
X=X.iloc[non_anom_index]
y=y.iloc[non_anom_index]

# =============================================================================
# #Final Gradient Boosting Model 
# =============================================================================

#Model With Parameters
GB=GradientBoostingClassifier(random_state=5,n_estimators=117,min_samples_leaf=11, min_samples_split=18,max_depth=6,max_features='sqrt',learning_rate=0.15) 
model2=GB.fit(X,y)
scores=cross_val_score(estimator=model2,X=X,y=y,cv=5)
cross_score=np.average(scores)


#kickstarter_grading_df = pd.read_excel("...ENTER PATH...\\Kickstarter-Grading.xlsx")


kickstarter_grading_df.drop( kickstarter_grading_df[ (kickstarter_grading_df['state'] !='failed') & (kickstarter_grading_df['state']!='successful')].index , inplace=True)

#conert goal to used
kickstarter_grading_df['goal_in_usd'] =kickstarter_grading_df['goal']*kickstarter_grading_df['static_usd_rate']

#drop unneccessary columns
kickstarter_grading_df=kickstarter_grading_df.drop(columns=['project_id','name','disable_communication','currency','pledged','usd_pledged','goal','static_usd_rate',
                    'deadline','created_at','state_changed_at','launched_at',
                    'state_changed_at_month', 'state_changed_at_day','state_changed_at_yr','state_changed_at_hr',
                    'backers_count','spotlight','staff_pick',
                    'state_changed_at_weekday','launch_to_state_change_days','launched_at_weekday'
           ])

kickstarter_grading_df.loc[kickstarter_grading_df.state =='successful', 'state'] = 0
kickstarter_grading_df.loc[kickstarter_grading_df.state =='failed', 'state'] = 1
kickstarter_grading_df=kickstarter_grading_df.dropna()

#specify x and y
y_grading=kickstarter_grading_df['state']
y_grading=y_grading.astype('int')

    
X_grading=kickstarter_grading_df.drop(columns=['state'])
X_grading=pd.get_dummies(X_grading,columns=['country','category','deadline_weekday','created_at_weekday'])


y_grading_pred = model2.predict(X_grading)

AS=accuracy_score(y_grading, y_grading_pred)
print("Score for Test set is : "+str(cross_score))
print('Score For Grading is is : '+ str(AS))

#CLASSIFICATION ENDS HERE


##############################################################################
#Clustering
##############################################################################


##############################################################################
#Data preprocessing
##############################################################################

#Importing the data
df=pd.read_excel('D:\\Downloads\\Kickstarter.xlsx')
#dropping observations
df.drop( df[ (df['state'] !='failed') & (df['state']!='successful')].index , inplace=True)
df.loc[df.state =='successful', 'state'] = 0
df.loc[df.state =='failed', 'state'] = 1
df.state=df.state.astype('int')

#creating a more scalable goal
df['goal_in_usd'] =df['goal']*df['static_usd_rate']
#dropping columns and NA
df=df.drop(columns=['goal','static_usd_rate'])
df=df.drop(columns='launch_to_state_change_days')
df=df.dropna()
df=df.drop(columns=['project_id','pledged'])

#we will be dealing with only numerical variables
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
X = df.select_dtypes(include=numerics)
# i will choose 3 variables
X_selected=df[['usd_pledged','create_to_launch_days','goal_in_usd']]
#Anomaly using Isolation Forest of each predictor alone:
iforest=IsolationForest(n_estimators=100,contamination=0.01,random_state=5)


pred1=iforest.fit_predict(X_selected)
score1=iforest.decision_function(X_selected)
non_anom_index1=where(pred1==1)
X_selected=X_selected.iloc[non_anom_index1]


##############################################################################
#K_Means_Clustering
##############################################################################
kmeans_scaler = StandardScaler()
X_std = kmeans_scaler.fit_transform(X_selected)

from sklearn.cluster import KMeans
withinss = []
for i in range(2,20):
    kmeans = KMeans(n_clusters=i)
    kmeansmodel = kmeans.fit(X_std)
    withinss.append(kmeansmodel.inertia_)

pyplot.plot([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], withinss)
plt.show()#4 is optimal

kmeans = KMeans(n_clusters=4,random_state=5)
model = kmeans.fit(X_std)
labels = model.predict(X_std)

X_with_clusters = pd.concat([X_selected.reset_index(drop=True),pd.DataFrame(labels, columns=["labels"])], axis=1)
cluster0 = X_with_clusters.loc[X_with_clusters["labels"]==0]
cluster1 = X_with_clusters.loc[X_with_clusters["labels"]==1]
cluster2 = X_with_clusters.loc[X_with_clusters["labels"]==2]
cluster3 = X_with_clusters.loc[X_with_clusters["labels"]==3]


cluster0 = cluster0.describe(include='all').transpose()
cluster1 = cluster1.describe(include='all').transpose()
cluster2 = cluster2.describe(include='all').transpose()
cluster3 = cluster3.describe(include='all').transpose()


#Plotly only on jupyter
 #fig = px.scatter_3d(X_selected, x='usd_pledged', y='goal_in_usd', z='create_to_launch_days',
               #color=labels)
 #fig.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_selected['goal_in_usd'], X_selected['usd_pledged'], X_selected['create_to_launch_days'],c=labels, cmap ='rainbow')
ax.set_xlabel('USD Pledged')
ax.set_ylabel('Goal in USD')
ax.set_zlabel('Create to launch Days')

print("Silhouette score is :")
print(silhouette_score(X_std, labels))
# 0.775
print("calinski_harabasz_score is :")
score = calinski_harabasz_score(X_std, labels)
print(score)
# 8551.06

print("pvalue is :")
df1 = 3 # df1=k-1
df2 = 14071-3 # df2=n-k
pvalue = 1-f.cdf(score, df1, df2)
print(pvalue)
#1.1102230246251565e-16