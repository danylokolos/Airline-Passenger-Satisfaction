# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 21:19:52 2021

@author: Danylo
"""
### Script to analyze customer satisfaction in Airlines



#%% Read in Data
import numpy as np
import pandas as pd
import pickle
import os

infile = "Data_PassengerSatisfaction.csv"
df = pd.read_csv(infile)




#%% Exploratory Data Analysis
# if error, try to run in Jupyter Notebook
#!pip install pandas-profiling

outfile = "PandasProfilingReport-AirlinePassengerSatisfaction.html"
if os.path.exists(outfile) == False:
    from pandas_profiling import ProfileReport
    profile = ProfileReport(df)
    profile.to_file(outfile)
    del profile
    del outfile





#%% Data Wrangling
# get rid of missing values in "Arrival Delay in Minutes"
df.shape
df = df.dropna(axis=0, how="any")
df.reset_index(inplace = True)
df.shape






#%% Preprocessing
# drop columns with useless info
df = df.drop(['Unnamed: 0','id'],axis=1)
df.columns

feature_names = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class',
       'Flight Distance', 'Inflight wifi service',
       'Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
target_names = ['satisfaction']


# encode categorical features
from sklearn.preprocessing import OrdinalEncoder
oenc = OrdinalEncoder(categories=[['Female', 'Male'],
                                  ['Loyal Customer', 'disloyal Customer'],
                                  ['Business travel', 'Personal Travel'],
                                  ['Business', 'Eco Plus', 'Eco']])
oenc.fit(df[['Gender', 'Customer Type','Type of Travel', 'Class']])
df[['Gender', 'Customer Type','Type of Travel', 'Class']] = oenc.transform(df[['Gender', 'Customer Type','Type of Travel', 'Class']])

# save ordinal encoder
oenc_filename= "oenc_target.pkl"
with open(oenc_filename, 'wb') as outfile:
    pickle.dump(oenc,outfile)


# encode categorical target
from sklearn.preprocessing import LabelEncoder
lenc = LabelEncoder()
_y = df[['satisfaction']].values
lenc.fit(_y.ravel())
#df[['satisfaction']] = lenc.transform(_y.ravel())
_a = pd.DataFrame(lenc.transform(_y.ravel()))
df[['satisfaction']] = _a

target_possibilities = df.satisfaction.unique()

# save label encoder
lenc_filename= "lenc_target.pkl"
with open(lenc_filename, 'wb') as outfile:
    pickle.dump(lenc,outfile)






#%% Split data
from sklearn.model_selection import train_test_split
X = df[feature_names]
y = df[target_names]

random_state = 42
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)



# Undersample Majority Class - Random - results in lower accuracy
"""
from imblearn.under_sampling import RandomUnderSampler
print('Length of X_train before Rebalance:', str(len(X_train)))
undersample = RandomUnderSampler(sampling_strategy='majority')
X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)
print('Length of X_train after Rebalance:', str(len(X_train_under)))

X_train = X_train_under
y_train = y_train_under
"""



#%% Machine Learning Model
from sklearn.ensemble import RandomForestClassifier

n_estimators = 100
random_state = 42
model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
model.fit(X_train,y_train.values.ravel())


# save model
model_filename = "model_RF.pkl"
with open(model_filename, 'wb') as outfile:
    pickle.dump(model,outfile)
    



#%% Run Model on Test Dataset
y_test_pred = model.predict(X_test)




#%% Analyze Results
# Feature Importance
feature_importance = list(zip(X_train, model.feature_importances_))
print('========== Feature Importance ==========')
print(*feature_importance, sep='\n')
print('========== END ==========')

# Accuracy Score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_test_pred)
print('========== Accuracy Score ==========')
print(accuracy)
print('========== END ==========')


# Confusion Matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)

print('========== Confusion Matrix ==========')
print(conf_matrix)
print('========== END ==========')




#%% Plot Confusion Matrix
import matplotlib.pyplot as plt

plt.figure(figsize = (10, 10))
cmap = plt.cm.Blues
plt.imshow(conf_matrix,cmap=cmap)
plt.grid(False)
plt.title('Customer Satisfaction Confusion Matrix', size = 24)
plt.colorbar(aspect=5)
output_labels = lenc.inverse_transform(target_possibilities)
tick_marks = np.arange(len(output_labels))
plt.xticks(tick_marks,output_labels,rotation=30,fontsize='xx-large')
plt.yticks(tick_marks,output_labels,fontsize='xx-large')
for ii in range(len(output_labels)):
    for jj in range(len(output_labels)):
        if conf_matrix[ii,jj] > np.max(conf_matrix)/2:
            plt.text(ii,jj,conf_matrix[ii,jj],horizontalalignment="center",color="white",fontsize='xx-large')
        else:
            plt.text(ii,jj,conf_matrix[ii,jj],horizontalalignment="center",fontsize='xx-large')
plt.tight_layout(pad=1)
plt.savefig('Plot_ConfusionMatrix.png')


#%% Plot Confusion Matrix Normalized
conf_matrix_norm = conf_matrix / conf_matrix.max()
plt.figure(figsize = (10, 10))
cmap = plt.cm.Blues
plt.imshow(conf_matrix_norm,cmap=cmap)
plt.grid(False)
plt.title('Customer Satisfaction Confusion Matrix Normalized', size = 24)
plt.colorbar(aspect=5)
output_labels = lenc.inverse_transform(target_possibilities)
tick_marks = np.arange(len(output_labels))
plt.xticks(tick_marks,output_labels,rotation=30,fontsize='xx-large')
plt.yticks(tick_marks,output_labels,fontsize='xx-large')
for ii in range(len(output_labels)):
    for jj in range(len(output_labels)):
        if conf_matrix_norm[ii,jj] > np.max(conf_matrix_norm)/2:
            plt.text(ii,jj,"{:.3f}".format(conf_matrix_norm[ii,jj]),horizontalalignment="center",color="white",fontsize='xx-large')
        else:
            plt.text(ii,jj,"{:.3f}".format(conf_matrix_norm[ii,jj]),horizontalalignment="center",fontsize='xx-large')
plt.tight_layout(pad=1)
plt.savefig('Plot_ConfusionMatrixNorm.png')



#%% Plot Precision-Recall Curve
from sklearn.metrics import PrecisionRecallDisplay

y_score = model.predict_proba(X_test)
y_score = y_score[:,1]
plt.figure()
disp = PrecisionRecallDisplay.from_predictions(y_test, y_score, name="Random Forest")
_ = disp.ax_.set_title('Precision Recall Curve')
plt.savefig('Plot_PrecisionRecallCurve.png')




#%% Plot ROC Curve
import sklearn.metrics as metrics

probs = model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.02, 1])
plt.ylim([0, 1.02])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('Plot_ROCCurve.png')
