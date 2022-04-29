# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 08:33:33 2022

@author: Acioli
"""
import pandas as pd

# GR

gr_features_selected = pd.read_csv("selected_features_data_gr.csv", low_memory=False)

gr_features_selected.pop('overall_survival')

gr_features_selected_column_names = list(gr_features_selected)

# RF

r_features_selected = pd.read_csv("selected_features_data_r.csv", low_memory=False)

r_features_selected.pop('overall_survival')

r_features_selected_column_names = list(r_features_selected)

# GR + RF

gr_r_features_selected = pd.read_csv("selected_features_data_gr_r.csv", low_memory=False)

gr_r_features_selected.pop('overall_survival')

gr_r_features_selected_column_names = list(gr_r_features_selected)