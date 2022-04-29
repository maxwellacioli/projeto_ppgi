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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

labelEncoder = LabelEncoder()

mlp = MLPClassifier(random_state=1, max_iter=300)
svc = make_pipeline(StandardScaler(), SVC(gamma='auto'))
dt = DecisionTreeClassifier(random_state=0)
neigh = KNeighborsClassifier(n_neighbors=25, leaf_size=10)
gnb = GaussianNB()
rf = RandomForestClassifier(criterion='entropy', max_depth= 7, class_weight='balanced', random_state=42)

###### UNSELECTED FEATURES #########

# df_treated = pd.read_csv("metabric_treated_data.csv", low_memory=False)

# df_treated = df_treated.iloc[: , 1:]

# target0 = df_treated.pop('overall_survival')

# df_treated_column_names = list(df_treated)

# for column in df_treated_column_names:
#     df_treated[column] = labelEncoder.fit_transform(df_treated[column])


###### SELECTED FEATURES - GAIN RATIO #########

gr_features_selected = pd.read_csv("selected_features_data_gr.csv", low_memory=False)

target_gr = gr_features_selected.pop('overall_survival')

gr_features_selected_column_names = list(gr_features_selected)

for column in gr_features_selected_column_names:
    gr_features_selected[column] = labelEncoder.fit_transform(gr_features_selected[column])
    
###### SELECTED FEATURES - RELIEF #########

r_features_selected = pd.read_csv("selected_features_data_r.csv", low_memory=False)

target_r = r_features_selected.pop('overall_survival')

r_features_selected_column_names = list(r_features_selected)

for column in r_features_selected_column_names:
    r_features_selected[column] = labelEncoder.fit_transform(r_features_selected[column])

###### SELECTED FEATURES - GAIN RATIO + RELIEF #########

gr_r_features_selected = pd.read_csv("selected_features_data_gr_r.csv", low_memory=False)

target_gr_r = gr_r_features_selected.pop('overall_survival')

gr_r_features_selected_column_names = list(gr_r_features_selected)

for column in gr_r_features_selected_column_names:
    gr_r_features_selected[column] = labelEncoder.fit_transform(gr_r_features_selected[column])    


###################################

columns_metadata_names = ["time_acc_rf_gr_feature_selection",
                "time_acc_rf_r_feature_selection",
                "time_acc_rf_gr_r_feature_selection",
                ]

metadata_df = pd.DataFrame()

index = 1
while index <= 50:
    print('current index: ', index)
    ## Accuracy ##
    
    # svc #
    
    clf = svc
    
    t = time()
    acc_svc_gr_feature_selection = np.mean(cross_val_score(clf, gr_features_selected, target_gr, cv=10, scoring='accuracy'))
    time_acc_svc_gr_feature_selection = round(time()-t,3)
  
    t = time()
    acc_svc_r_feature_selection = np.mean(cross_val_score(clf, r_features_selected, target_r, cv=10, scoring='accuracy'))
    time_acc_svc_r_feature_selection = round(time()-t,3)
    
    t = time()
    acc_svc_gr_r_feature_selection = np.mean(cross_val_score(clf, gr_r_features_selected, target_gr_r, cv=10, scoring='accuracy'))
    time_acc_svc_gr_r_feature_selection = round(time()-t,3)
    
    # mlp #
    
    clf = mlp
    
    t = time()
    acc_mlp_gr_feature_selection = np.mean(cross_val_score(clf, gr_features_selected, target_gr, cv=10, scoring='accuracy'))
    time_acc_mlp_gr_feature_selection = round(time()-t,3)
  
    t = time()
    acc_mlp_r_feature_selection = np.mean(cross_val_score(clf, r_features_selected, target_r, cv=10, scoring='accuracy'))
    time_acc_mlp_r_feature_selection = round(time()-t,3)
    
    t = time()
    acc_mlp_gr_r_feature_selection = np.mean(cross_val_score(clf, gr_r_features_selected, target_gr_r, cv=10, scoring='accuracy'))
    time_acc_mlp_gr_r_feature_selection = round(time()-t,3)
    

    # knn#
    
    clf = neigh

    t = time()
    acc_knn_gr_feature_selection = np.mean(cross_val_score(clf, gr_features_selected, target_gr, cv=10, scoring='accuracy'))
    time_acc_knn_gr_feature_selection = round(time()-t,3)
  
    t = time()
    acc_knn_r_feature_selection = np.mean(cross_val_score(clf, r_features_selected, target_r, cv=10, scoring='accuracy'))
    time_acc_knn_r_feature_selection = round(time()-t,3)
    
    t = time()
    acc_knn_gr_r_feature_selection = np.mean(cross_val_score(clf, gr_r_features_selected, target_gr_r, cv=10, scoring='accuracy'))
    time_acc_knn_gr_r_feature_selection = round(time()-t,3)
    
    # gnb #
    
    clf = gnb
    
    t = time()
    acc_gnb_gr_feature_selection = np.mean(cross_val_score(clf, gr_features_selected, target_gr, cv=10, scoring='accuracy'))
    time_acc_gnb_gr_feature_selection = round(time()-t,3)

    t = time()
    acc_gnb_r_feature_selection = np.mean(cross_val_score(clf, r_features_selected, target_r, cv=10, scoring='accuracy'))
    time_acc_gnb_r_feature_selection = round(time()-t,3)
    
    t = time()
    acc_gnb_gr_r_feature_selection = np.mean(cross_val_score(clf, gr_r_features_selected, target_gr_r, cv=10, scoring='accuracy'))
    time_acc_gnb_gr_r_feature_selection = round(time()-t,3)
    
    # rf #
    
    clf = rf
    
    t = time()
    acc_rf_gr_feature_selection = np.mean(cross_val_score(clf, gr_features_selected, target_gr, cv=10, scoring='accuracy'))
    time_acc_rf_gr_feature_selection = round(time()-t,3)

    t = time()
    acc_rf_r_feature_selection = np.mean(cross_val_score(clf, r_features_selected, target_r, cv=10, scoring='accuracy'))
    time_acc_rf_r_feature_selection = round(time()-t,3)
    
    t = time()
    acc_rf_gr_r_feature_selection = np.mean(cross_val_score(clf, gr_r_features_selected, target_gr_r, cv=10, scoring='accuracy'))
    time_acc_rf_gr_r_feature_selection = round(time()-t,3)
    
#     ## Precision ##
    
#     # svc #
    
#     clf = svc
    
#     t = time()
#     prec_svc_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='precision'))
#     time_prec_svc_original_feats = round(time()-t,3)

#     t = time()
#     prec_svc_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='precision'))
#     time_prec_svc_selected_feats = round(time()-t,3)
    
#      # mlp #
    
#     clf = mlp
    
#     t = time()
#     prec_mlp_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='precision'))
#     time_prec_mlp_original_feats = round(time()-t,3)

#     t = time()
#     prec_mlp_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='precision'))
#     time_prec_mlp_selected_feats = round(time()-t,3)
    
    
#     # dt #

#     clf = dt
    
#     t = time()
#     prec_dt_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='precision'))
#     time_prec_dt_original_feats = round(time()-t,3)
    
#     t = time()
#     prec_dt_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='precision'))
#     time_prec_dt_selected_feats = round(time()-t,3)
    
#     # knn#
    
#     clf = neigh
    
#     t = time()
#     prec_neight_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='precision'))
#     time_prec_neight_original_feats = round(time()-t,3)
    
#     t = time()
#     prec_neight_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='precision'))
#     time_prec_neight_selected_feats = round(time()-t,3)
    
#     # gnb #
    
#     clf = gnb
    
#     t = time()
#     prec_gnb_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='precision'))
#     time_prec_gnb_original_feats = round(time()-t,3)
    
#     t = time()
#     prec_gnb_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='precision'))
#     time_prec_gnb_selected_feats = round(time()-t,3)
    
#       # rf #
    
#     clf = rf
    
#     t = time()
#     prec_rf_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='precision'))
#     time_prec_rf_original_feats = round(time()-t,3)
    
#     t = time()
#     prec_rf_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='precision'))
#     time_prec_rf_selected_feats = round(time()-t,3)
    
#     ## Recall ##
    
#     # svc #
    
#     clf = svc
    
#     t = time()
#     rec_svc_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='recall'))
#     time_rec_svc_original_feats = round(time()-t,3)

#     t = time()
#     rec_svc_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='recall'))
#     time_rec_svc_selected_feats = round(time()-t,3)
    
#      # mlp #
    
#     clf = mlp
    
#     t = time()
#     rec_mlp_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='recall'))
#     time_rec_mlp_original_feats = round(time()-t,3)

#     t = time()
#     rec_mlp_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='recall'))
#     time_rec_mlp_selected_feats = round(time()-t,3)
    
    
#     # dt #

#     clf = dt
    
#     t = time()
#     rec_dt_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='recall'))
#     time_rec_dt_original_feats = round(time()-t,3)
    
#     t = time()
#     rec_dt_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='recall'))
#     time_rec_dt_selected_feats = round(time()-t,3)
    
#     # knn#
    
#     clf = neigh
    
#     t = time()
#     rec_neight_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='recall'))
#     time_rec_neight_original_feats = round(time()-t,3)
    
#     t = time()
#     rec_neight_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='recall'))
#     time_rec_neight_selected_feats = round(time()-t,3)
    
#     # gnb #
    
#     clf = gnb
    
#     t = time()
#     rec_gnb_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='recall'))
#     time_rec_gnb_original_feats = round(time()-t,3)
    
#     t = time()
#     rec_gnb_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='recall'))
#     time_rec_gnb_selected_feats = round(time()-t,3)
    
#       # rf #
    
#     clf = rf
    
#     t = time()
#     rec_rf_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='recall'))
#     time_rec_rf_original_feats = round(time()-t,3)
    
#     t = time()
#     rec_rf_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='recall'))
#     time_rec_rf_selected_feats = round(time()-t,3)
    
#     ## F1 ##
    
#     # svc #
    
#     clf = svc
    
#     t = time()
#     f1_svc_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='f1'))
#     time_f1_svc_original_feats = round(time()-t,3)

#     t = time()
#     f1_svc_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='f1'))
#     time_f1_svc_selected_feats = round(time()-t,3)
    
#      # mlp #
    
#     clf = mlp
    
#     t = time()
#     f1_mlp_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='f1'))
#     time_f1_mlp_original_feats = round(time()-t,3)

#     t = time()
#     f1_mlp_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='f1'))
#     time_f1_mlp_selected_feats = round(time()-t,3)
    
    
#     # dt #

#     clf = dt
    
#     t = time()
#     f1_dt_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='f1'))
#     time_f1_dt_original_feats = round(time()-t,3)
    
#     t = time()
#     f1_dt_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='f1'))
#     time_f1_dt_selected_feats = round(time()-t,3)
    
#     # knn#
    
#     clf = neigh
    
#     t = time()
#     f1_neight_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='f1'))
#     time_f1_neight_original_feats = round(time()-t,3)
    
#     t = time()
#     f1_neight_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='f1'))
#     time_f1_neight_selected_feats = round(time()-t,3)
    
#     # gnb #
    
#     clf = gnb
    
#     t = time()
#     f1_gnb_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='f1'))
#     time_f1_gnb_original_feats = round(time()-t,3)
    
#     t = time()
#     f1_gnb_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='f1'))
#     time_f1_gnb_selected_feats = round(time()-t,3)
    
#       # rf #
    
#     clf = rf
    
#     t = time()
#     f1_rf_original_feats = np.mean(cross_val_score(clf, df_treated, target0, cv=10, scoring='f1'))
#     time_f1_rf_original_feats = round(time()-t,3)
    
#     t = time()
#     f1_rf_selected_feats = np.mean(cross_val_score(clf, df_treated_selected, target1, cv=10, scoring='f1'))
#     time_f1_rf_selected_feats = round(time()-t,3)
    
    new_data = {
                "time_acc_rf_gr": time_acc_rf_gr_feature_selection, 
                "time_acc_rf_r": time_acc_rf_r_feature_selection,
                "time_acc_rf_gr_r": time_acc_rf_gr_r_feature_selection,
                "time_acc_gnb_gr": time_acc_gnb_gr_feature_selection, 
                "time_acc_gnb_r": time_acc_gnb_r_feature_selection,
                "time_acc_gnb_gr_r": time_acc_gnb_gr_r_feature_selection,
                "time_acc_mlp_gr": time_acc_mlp_gr_feature_selection, 
                "time_acc_mlp_r": time_acc_mlp_r_feature_selection,
                "time_acc_mlp_gr_r": time_acc_mlp_gr_r_feature_selection,
                "time_acc_svc_gr": time_acc_svc_gr_feature_selection, 
                "time_acc_svc_r": time_acc_svc_r_feature_selection,
                "time_acc_svc_gr_r": time_acc_svc_gr_r_feature_selection,
                "time_acc_knn_gr": time_acc_knn_gr_feature_selection, 
                "time_acc_knn_r": time_acc_knn_r_feature_selection,
                "time_acc_knn_gr_r": time_acc_knn_gr_r_feature_selection,
                }
    
    metadata_df = metadata_df.append(new_data, ignore_index=True)
    
    index+=1

metadata_df.to_csv('metadata_acc.csv', sep=';', encoding='utf-8', index=False)
