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


### ML



### Results