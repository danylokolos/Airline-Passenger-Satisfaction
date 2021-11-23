# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 21:19:52 2021

@author: Danylo
"""

### Read in Data
import pandas as pd
import pickle

infile = "Data_PassengerSatisfaction.csv"
df = pd.read_csv(infile)



### Data Wrangling
# none needed,



### Preprocessing
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
df[['satisfaction']] = lenc.transform(_y.ravel())

# save label encoder
lenc_filename= "lenc_target.pkl"
with open(lenc_filename, 'wb') as outfile:
    pickle.dump(lenc,outfile)



### Split data
from sklearn.model_selection import train_test_split
X = df[feature_names]
y = df[target_names]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



### ML



### Results