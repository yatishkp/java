import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler


from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

# Reading, Removing Sate and Phone cloumns from the dataset.

data = pd.read_csv('/home/yatish/Documents/all-docs/python/telecom/telecom_churn_data.csv')
data = data.iloc[:,1:]
#print(data.head())
data = data.drop('Phone',axis = 1)
#print(data)

# Data Preprocessing  Label Encoding for International Plan and Vmail Plan.
labelencoder_X = LabelEncoder()
data.iloc[:, 2] = labelencoder_X.fit_transform(data.iloc[:, 2])
data.iloc[:, 3] = labelencoder_X.fit_transform(data.iloc[:, 3])
print(data.head())
data.iloc[:, -1] = labelencoder_X.fit_transform(data.iloc[:, -1])
scaler = MinMaxScaler()
data[['A', 'B']] = scaler.fit_transform(data[['A', 'B']])
print(data.Churn.value_counts())

#X = data.iloc[:,0:17]  #independent columns
#y = data.iloc[:,-1]    #target column i.e price range
#apply SelectKBest class to extract top 10 best features
'''bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(10,'Score'))'''

corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
print('Done')

# we compare the correlation between features and remove one of two features that have a correlation higher than 0.9

columns = np.full((corrmat.shape[0],), True, dtype=bool)
for i in range(corrmat.shape[0]):
    for j in range(i+1, corrmat.shape[0]):
        if corrmat.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = data.columns[columns]
#print(selected_columns)
data = data[selected_columns]
#print(data.shape)

X = data.iloc[:,0:12]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#print(dfcolumns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
#print(featureScores.nlargest(10,'Score'))

feature_cols = ['Day Mins', 'Eve Mins', 'International Plan', 'Night Mins',
       'Vmail Plan', 'Account Lengh', 'International Mins', 'International Calls', 'Day Calls',
       'Area Code','Churn']
df1 = pd.DataFrame(data, columns=feature_cols)
#print(df1.shape)

X = df1.iloc[:,0:10]
Y = df1.iloc[:,-1]

print(X.head())
print(Y.head())

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=10)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy_DecisionTree:",metrics.accuracy_score(y_test, y_pred))

model = XGBClassifier(learning_rate =0.01,
                      gamma=0,n_estimators=1000,
                     random_state=25,
                    min_child_weight=2,
                     max_depth=4,
                     reg_alpha=0.2,
                      objective= 'binary:logistic',
                        subsample=0.8,
                      colsample_bytree=0.8,
                      nthread=4,scale_pos_weight=1,seed=27)

model.fit(X_train, y_train)
y_predict = model.predict(X_test)

# Instantiate model with 1000 decision trees
'''rf = RandomForestClassifier(n_estimators=2000,
                               bootstrap = True,
                               max_features = 'sqrt')
# Train the model on training data
clf_rand = rf.fit(X_train,y_train);

y_pred_random = clf_rand.predict(X_test)'''
matrix = confusion_matrix(y_test,y_predict)
print(matrix)
#y_test['churn'] = y_test
y_test.to_csv('test.csv')
print("Accuracy_RandomForest:",metrics.accuracy_score(y_test, y_pred_random))

'''feature_cols_graph = ['Day Mins', 'Eve Mins', 'International Plan', 'Night Mins',
       'Vmail Plan', 'Account Lengh', 'International Mins', 'International Calls', 'Day Calls',
       'Area Code']

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols_graph,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())'''

rslt_df = df1[df1['Churn']==1]
#print(rslt_df)

avg_yes = rslt_df.mean(axis = 1)
print(avg_yes)







