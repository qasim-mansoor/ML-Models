import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error
import sklearn.metrics as metrics

from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv'

# await download(path, "Weather_Data.csv")
filename ="Weather_Data.csv"

df = pd.read_csv("Weather_Data.csv")

df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])

df_sydney_processed.replace(['No', 'Yes'], [0,1], inplace=True)

df_sydney_processed.drop('Date',axis=1,inplace=True)
df_sydney_processed = df_sydney_processed.astype(float)
features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
Y = df_sydney_processed['RainTomorrow']

x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.2, random_state=10)

#                       Linear Regression         #

LinearReg = LinearRegression().fit(x_train, y_train)
predictions = LinearReg.predict(x_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LinearRegression_MAE = mean_absolute_error(y_test, predictions)
LinearRegression_MSE = mean_squared_error(y_test, predictions)
LinearRegression_R2 = r2_score(y_test, predictions)
metrics = {
    'Metric': ['MAE', 'MSE', 'R2'],
    'Value': [LinearRegression_MAE, LinearRegression_MSE, LinearRegression_R2]
}

Report = pd.DataFrame(metrics)
print(Report)

#                         KNN                               #

KNN = KNeighborsClassifier(n_neighbors=4).fit(x_train,y_train)
predictions = KNN.predict(x_test)
KNN_Accuracy_Score = accuracy_score(y_test, predictions)
KNN_JaccardIndex = jaccard_score(y_test, predictions)
KNN_F1_Score = f1_score(y_test, predictions)

print("Accuracy Score: %.2f" % KNN_Accuracy_Score)
print("Jaccard Index Score: %.2f" % KNN_JaccardIndex)
print("F1 Score: %.2f" % KNN_F1_Score)

#                         Decision Tree                               #

Tree = DecisionTreeClassifier().fit(x_train, y_train)
predictions = Tree.predict(x_test)
Tree_Accuracy_Score = accuracy_score(y_test, predictions)
Tree_JaccardIndex = jaccard_score(y_test, predictions)
Tree_F1_Score = f1_score(y_test, predictions)

print("Accuracy Score: %.2f" % Tree_Accuracy_Score)
print("Jaccard Index Score: %.2f" % Tree_JaccardIndex)
print("F1 Score: %.2f" % Tree_F1_Score)

#                         Logistic Regression                               #

x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.2, random_state=1)
LR = LogisticRegression(solver="liblinear").fit(x_train, y_train)
predictions = LR.predict(x_test)
predict_proba = LR.predict_proba(x_test)
LR_Accuracy_Score = accuracy_score(y_test, predictions)
LR_JaccardIndex = jaccard_score(y_test, predictions)
LR_F1_Score = f1_score(y_test, predictions)
LR_Log_Loss = log_loss(y_test, predict_proba)

print("Accuracy Score: %.2f" % LR_Accuracy_Score)
print("Jaccard Index Score: %.2f" % LR_JaccardIndex)
print("F1 Score: %.2f" % LR_F1_Score)
print("Log Loss: %.2f" % LR_Log_Loss)

#                         SVM                               #

SVM = svm.SVC().fit(x_train, y_train)
predictions = SVM.predict(x_test)
SVM_Accuracy_Score = accuracy_score(y_test, predictions)
SVM_JaccardIndex = jaccard_score(y_test, predictions)
SVM_F1_Score = f1_score(y_test, predictions)

print("Accuracy Score: %.2f" % SVM_Accuracy_Score)
print("Jaccard Index Score: %.2f" % SVM_JaccardIndex)
print("F1 Score: %.2f" % SVM_F1_Score)


#                                   Final Report                  #

Report = [
    {
        'Model': 'KNN',
        'Accuracy_Score': KNN_Accuracy_Score,
        'JaccardIndex': KNN_JaccardIndex,
        'F1_Score': KNN_F1_Score
    },
    {
        'Model': 'Decision Tree',
        'Accuracy_Score': Tree_Accuracy_Score,
        'JaccardIndex': Tree_JaccardIndex,
        'F1_Score': Tree_F1_Score
    },
    {
        'Model': 'Logistic Regression',
        'Accuracy_Score': LR_Accuracy_Score,
        'JaccardIndex': LR_JaccardIndex,
        'F1_Score': LR_F1_Score,
        'Log Loss': LR_Log_Loss
    },
    {
        'Model': 'SVM',
        'Accuracy_Score': SVM_Accuracy_Score,
        'JaccardIndex': SVM_JaccardIndex,
        'F1_Score': SVM_F1_Score,
    }
]

df = pd.DataFrame(Report)
print(df)
