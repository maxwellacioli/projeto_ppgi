# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 07:18:02 2022

@author: Acioli
"""

from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from time import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

###### SELECTED FEATURES #########

labelEncoder = LabelEncoder()

df = pd.read_csv("new_selected_features_data.csv", low_memory=False)

target = df.pop('overall_survival')

df_column_names = list(df)

for column in df_column_names:
    df[column] = labelEncoder.fit_transform(df[column])
    
    
X_selected_train, X_selected_test, y_selected_train, y_selected_test = train_test_split(df, 
                                                                                        target, test_size=0.2, 
                                                                                        random_state=42)            