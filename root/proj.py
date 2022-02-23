# -*- coding: utf-8 -*-
"""
Created on Sun Feb 6 20:11:30 2022

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

labelEncoder = LabelEncoder()

lr = LogisticRegression(random_state=50,max_iter=10000)
neigh = KNeighborsClassifier(n_neighbors=25, leaf_size=10)
gnb = GaussianNB()
rf = RandomForestClassifier(criterion='entropy', max_depth= 7, class_weight='balanced', random_state=42)

###### UNSELECTED FEATURES #########

df_treated = pd.read_csv("metabric_treated_data.csv", low_memory=False)

df_treated = df_treated.iloc[: , 1:]

target0 = df_treated.pop('type_of_breast_surgery')

df_treated_column_names = list(df_treated)

for column in df_treated_column_names:
    df_treated[column] = labelEncoder.fit_transform(df_treated[column])


###### SELECTED FEATURES #########

df_treated_selected = pd.read_csv("selected_features_data.csv", low_memory=False)

target1 = df_treated_selected.pop('type_of_breast_surgery')

df_treated_selected_column_names = list(df_treated_selected)

for column in df_treated_selected_column_names:
    df_treated_selected[column] = labelEncoder.fit_transform(df_treated_selected[column])


###################################

columns_metadata_names = ["time_acc_lr_original_feats",
                "time_acc_lr_selected_feats",
                "time_acc_neight_original_feats",
                "time_acc_neight_selected_feats",
                "time_acc_gnb_original_feats",
                "time_acc_gnb_selected_feats",
                "time_acc_rf_original_feats",
                "time_acc_rf_selected_feats",
                "time_prec_lr_original_feats",
                "time_prec_lr_selected_feats",
                "time_prec_neight_original_feats",
                "time_prec_neight_selected_feats",
                "time_prec_gnb_original_feats",
                "time_prec_gnb_selected_feats",
                "time_prec_rf_original_feats",
                "time_prec_rf_selected_feats",
                "time_rec_lr_original_feats",
                "time_rec_lr_selected_feats",
                "time_rec_neight_original_feats",
                "time_rec_neight_selected_feats",
                "time_rec_gnb_original_feats",
                "time_rec_gnb_selected_feats",
                "time_rec_rf_original_feats",
                "time_rec_rf_selected_feats",
                "time_f1_lr_original_feats",
                "time_f1_lr_selected_feats",
                "time_f1_neight_original_feats",
                "time_f1_neight_selected_feats",
                "time_f1_gnb_original_feats",
                "time_f1_gnb_selected_feats",
                "time_f1_rf_original_feats",
                "time_f1_rf_selected_feats"]

metadata_df = pd.DataFrame(columns = columns_metadata_names)

index = 0
while index < 100:
    ## Accuracy ##
    
    # lr #
    
    clf = lr
    
    t = time()
    acc_lr_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='accuracy'))
    time_acc_lr_original_feats = round(time()-t,3)

    t = time()
    acc_lr_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='accuracy'))
    time_acc_lr_selected_feats = round(time()-t,3)
    
    # knn#
    
    clf = neigh
    
    t = time()
    acc_neight_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='accuracy'))
    time_acc_neight_original_feats = round(time()-t,3)

    t = time()
    acc_neight_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='accuracy'))
    time_acc_neight_selected_feats = round(time()-t,3)
    
    # gnb #
    
    clf = gnb
    
    t = time()
    acc_gnb_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='accuracy'))
    time_acc_gnb_original_feats = round(time()-t,3)

    t = time()
    acc_gnb_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='accuracy'))
    time_acc_gnb_selected_feats = round(time()-t,3)
    
      # rf #
    
    clf = rf
    
    t = time()
    acc_rf_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='accuracy'))
    time_acc_rf_original_feats = round(time()-t,3)

    t = time()
    acc_rf_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='accuracy'))
    time_acc_rf_selected_feats = round(time()-t,3)
    
    ## Precision ##
    
    # lr #

    clf = lr
    
    t = time()
    prec_lr_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='precision'))
    time_prec_lr_original_feats = round(time()-t,3)
    
    t = time()
    prec_lr_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='precision'))
    time_prec_lr_selected_feats = round(time()-t,3)
    
    # knn#
    
    clf = neigh
    
    t = time()
    prec_neight_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='precision'))
    time_prec_neight_original_feats = round(time()-t,3)
    
    t = time()
    prec_neight_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='precision'))
    time_prec_neight_selected_feats = round(time()-t,3)
    
    # gnb #
    
    clf = gnb
    
    t = time()
    prec_gnb_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='precision'))
    time_prec_gnb_original_feats = round(time()-t,3)
    
    t = time()
    prec_gnb_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='precision'))
    time_prec_gnb_selected_feats = round(time()-t,3)
    
      # rf #
    
    clf = rf
    
    t = time()
    prec_rf_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='precision'))
    time_prec_rf_original_feats = round(time()-t,3)
    
    t = time()
    prec_rf_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='precision'))
    time_prec_rf_selected_feats = round(time()-t,3)
    
    ## Recall ##
    
    # lr #

    clf = lr
    
    t = time()
    rec_lr_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='recall'))
    time_rec_lr_original_feats = round(time()-t,3)
    
    t = time()
    rec_lr_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='recall'))
    time_rec_lr_selected_feats = round(time()-t,3)
    
    # knn#
    
    clf = neigh
    
    t = time()
    rec_neight_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='recall'))
    time_rec_neight_original_feats = round(time()-t,3)
    
    t = time()
    rec_neight_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='recall'))
    time_rec_neight_selected_feats = round(time()-t,3)
    
    # gnb #
    
    clf = gnb
    
    t = time()
    rec_gnb_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='recall'))
    time_rec_gnb_original_feats = round(time()-t,3)
    
    t = time()
    rec_gnb_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='recall'))
    time_rec_gnb_selected_feats = round(time()-t,3)
    
      # rf #
    
    clf = rf
    
    t = time()
    rec_rf_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='recall'))
    time_rec_rf_original_feats = round(time()-t,3)
    
    t = time()
    rec_rf_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='recall'))
    time_rec_rf_selected_feats = round(time()-t,3)
    
    ## F1 ##
    
    # lr #

    clf = lr
    
    t = time()
    f1_lr_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='f1'))
    time_f1_lr_original_feats = round(time()-t,3)
    
    t = time()
    f1_lr_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='f1'))
    time_f1_lr_selected_feats = round(time()-t,3)
    
    # knn#
    
    clf = neigh
    
    t = time()
    f1_neight_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='f1'))
    time_f1_neight_original_feats = round(time()-t,3)
    
    t = time()
    f1_neight_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='f1'))
    time_f1_neight_selected_feats = round(time()-t,3)
    
    # gnb #
    
    clf = gnb
    
    t = time()
    f1_gnb_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='f1'))
    time_f1_gnb_original_feats = round(time()-t,3)
    
    t = time()
    f1_gnb_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='f1'))
    time_f1_gnb_selected_feats = round(time()-t,3)
    
      # rf #
    
    clf = rf
    
    t = time()
    f1_rf_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='f1'))
    time_f1_rf_original_feats = round(time()-t,3)
    
    t = time()
    f1_rf_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='f1'))
    time_f1_rf_selected_feats = round(time()-t,3)
    
    new_data = {"time_acc_lr_original_feats": time_acc_lr_original_feats, 
                "time_acc_lr_selected_feats": time_acc_lr_selected_feats,
                "time_acc_neight_original_feats": time_acc_neight_original_feats,
                "time_acc_neight_selected_feats": time_acc_neight_selected_feats,
                "time_acc_gnb_original_feats": time_acc_gnb_original_feats,
                "time_acc_gnb_selected_feats": time_acc_gnb_selected_feats,
                "time_acc_rf_original_feats": time_acc_rf_original_feats,
                "time_acc_rf_selected_feats": time_acc_rf_selected_feats,
                "time_prec_lr_original_feats": time_prec_lr_original_feats,
                "time_prec_lr_selected_feats": time_prec_lr_selected_feats,
                "time_prec_neight_original_feats": time_prec_neight_original_feats,
                "time_prec_neight_selected_feats": time_prec_neight_selected_feats,
                "time_prec_gnb_original_feats": time_prec_gnb_original_feats,
                "time_prec_gnb_selected_feats": time_prec_gnb_selected_feats,
                "time_prec_rf_original_feats": time_prec_rf_original_feats,
                "time_prec_rf_selected_feats": time_prec_rf_selected_feats,
                "time_rec_lr_original_feats": time_rec_lr_original_feats,
                "time_rec_lr_selected_feats": time_rec_lr_selected_feats,
                "time_rec_neight_original_feats": time_rec_neight_original_feats,
                "time_rec_neight_selected_feats": time_rec_neight_selected_feats,
                "time_rec_gnb_original_feats": time_rec_gnb_original_feats,
                "time_rec_gnb_selected_feats": time_rec_gnb_selected_feats,
                "time_rec_rf_original_feats": time_rec_rf_original_feats,
                "time_rec_rf_selected_feats": time_rec_rf_selected_feats,
                "time_f1_lr_original_feats": time_f1_lr_original_feats,
                "time_f1_lr_selected_feats": time_f1_lr_selected_feats,
                "time_f1_neight_original_feats": time_f1_neight_original_feats,
                "time_f1_neight_selected_feats": time_f1_neight_selected_feats,
                "time_f1_gnb_original_feats": time_f1_gnb_original_feats,
                "time_f1_gnb_selected_feats": time_f1_gnb_selected_feats,
                "time_f1_rf_original_feats": time_f1_rf_original_feats,
                "time_f1_rf_selected_feats": time_f1_rf_selected_feats}
    
    metadata_df = metadata_df.append(new_data, ignore_index=True)
    
    index+=1

metadata_df.to_csv('metadata9.csv', sep=';', encoding='utf-8', index=False)
