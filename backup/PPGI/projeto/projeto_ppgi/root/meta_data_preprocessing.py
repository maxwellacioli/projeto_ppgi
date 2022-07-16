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

###### SELECTED FEATURES #########

df_treated_selected = pd.read_csv("selected_features_data.csv", low_memory=False)

# target1 = df_treated_selected.pop('type_of_breast_surgery')

# df_treated_selected_column_names = list(df_treated_selected)

# for column in df_treated_selected_column_names:
#     df_treated_selected[column] = labelEncoder.fit_transform(df_treated_selected[column])