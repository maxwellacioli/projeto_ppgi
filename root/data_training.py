# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 14:23:00 2022

@author: Acioli
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 09 18:07:15 2022

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

labelEncoder = LabelEncoder()

nn = MLPClassifier(solver='lbfgs', alpha=1e-5, 
                   hidden_layer_sizes=(5, 2), random_state=1)
svc = SVC(gamma='auto', C=0.5)
lr = LogisticRegression(random_state=50,max_iter=10000)
neigh = KNeighborsClassifier(n_neighbors=25, leaf_size=10)
gnb = GaussianNB()
rf = RandomForestClassifier(criterion='entropy', max_depth= 7, class_weight='balanced', random_state=42)

###### UNSELECTED FEATURES #########

df_treated = pd.read_csv("metabric_treated_data.csv", low_memory=False)

df_treated = df_treated.iloc[: , 1:]

original_target = df_treated.pop('type_of_breast_surgery')

df_treated_column_names = list(df_treated)

for column in df_treated_column_names:
    df_treated[column] = labelEncoder.fit_transform(df_treated[column])
    
X_original_train, X_original_test, y_original_train, y_original_test = train_test_split(df_treated, 
                                                                                        original_target, test_size=0.2, 
                                                                                        random_state=42)    


###### SELECTED FEATURES #########

df_treated_selected = pd.read_csv("selected_features_data.csv", low_memory=False)

selected_target = df_treated_selected.pop('type_of_breast_surgery')

df_treated_selected_column_names = list(df_treated_selected)

for column in df_treated_selected_column_names:
    df_treated_selected[column] = labelEncoder.fit_transform(df_treated_selected[column])
    
X_selected_train, X_selected_test, y_selected_train, y_selected_test = train_test_split(df_treated_selected, 
                                                                                        selected_target, test_size=0.2, 
                                                                                        random_state=42)        


###################################

columns_metadata_names = ["time_lr_original_feats",
                          "time_lr_selected_feats",
                          "time_gnb_original_feats",
                          "time_gnb_selected_feats",
                          "time_rf_original_feats",
                          "time_rf_selected_feats",
                          "time_svc_original_feats",
                          "time_svc_selected_feats",
                          "time_nn_original_feats",
                          "time_nn_selected_feats"]

metadata_df = pd.DataFrame(columns = columns_metadata_names)

index = 0
while index < 10:
    ## Accuracy ##
    
    # lr #
    
    clf = lr
    
    t1 = time()
    clf.fit(X_original_train, y_original_train)
    time_lr_original_feats = round(time()-t1,3)

    t2 = time()
    clf.fit(X_selected_train, y_selected_train)
    time_lr_selected_feats = round(time()-t2,3)
    
    # gnb #
    
    clf = gnb
    
    t5 = time()
    clf.fit(X_original_train, y_original_train)
    time_gnb_original_feats = round(time()-t5,3)

    t6 = time()
    clf.fit(X_selected_train, y_selected_train)
    time_gnb_selected_feats = round(time()-t6,3)
    
    # rf #
    
    clf = lr
    
    t7 = time()
    clf.fit(X_original_train, y_original_train)
    time_rf_original_feats = round(time()-t7,3)

    t8 = time()
    clf.fit(X_selected_train, y_selected_train)
    time_rf_selected_feats = round(time()-t8,3)
    
    # svc #
    
    clf = lr
    
    t9 = time()
    clf.fit(X_original_train, y_original_train)
    time_svc_original_feats = round(time()-t9,3)

    t10 = time()
    clf.fit(X_selected_train, y_selected_train)
    time_svc_selected_feats = round(time()-t10,3)
    
     # neutral network #
    
    clf = lr
    
    t11 = time()
    clf.fit(X_original_train, y_original_train)
    time_nn_original_feats = round(time()-t11,3)

    t12 = time()
    clf.fit(X_selected_train, y_selected_train)
    time_nn_selected_feats = round(time()-t12,3)

    
    new_data = {"time_lr_original_feats": time_lr_original_feats, "time_lr_selected_feats": time_lr_selected_feats, 
                "time_gnb_original_feats": time_gnb_original_feats,"time_gnb_selected_feats": time_gnb_selected_feats,
                "time_rf_original_feats": time_rf_original_feats,"time_rf_selected_feats": time_rf_selected_feats, 
                "time_svc_original_feats": time_svc_original_feats,"time_svc_selected_feats": time_svc_selected_feats,
                "time_nn_original_feats": time_nn_original_feats,"time_nn_selected_feats": time_nn_selected_feats }
    
    metadata_df = metadata_df.append(new_data, ignore_index=True)
    
    index+=1

metadata_df.to_csv('metadata_3.csv', sep=';', encoding='utf-8', index=False)
