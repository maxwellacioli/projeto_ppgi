# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 15:19:24 2022

@author: Acioli
"""

# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from statsmodels.graphics.gofplots import qqplot
from sklearn import preprocessing
import seaborn as sns


# df_treated = pd.read_csv("metabric_treated_data.csv", low_memory=False)
# df_treated_column_names = list(df_treated)

# df_0 = pd.read_csv("metadata1.csv", sep=';')
# df_1 = pd.read_csv("metadata2.csv", sep=';')
df = pd.read_csv("metadata_acc.csv", sep=';')

df = df.rename(columns={df.columns[0]: 'time_acc_rf_gr'})
df = df.rename(columns={df.columns[1]: 'time_acc_rf_r'})
df = df.rename(columns={df.columns[2]: 'time_acc_rf_gr_r'})

# df_selected_feats = pd.read_csv("selected_features_data.csv", sep=',')
# columns_names_selected_feats = list(df_selected_feats)

#### HISTOGRAMS

# df.hist(figsize=(100, 60))

#### NEW DATA ####

# df = df.append(df_0, ignore_index=True)
# df = df.append(df_1, ignore_index=True)

columns_names = list(df)


# #### ACCURACY ####

# sns.distplot(df['time_acc_rf_original_feats'], hist = False, kde = True,
#                   kde_kws = {'shade': False, 'linewidth': 1}, 
#                   label = 'original_feats', color='red')

# sns.distplot(df['time_acc_rf_selected_feats'], hist = False, kde = True,
#                   kde_kws = {'shade': False, 'linewidth': 1}, 
#                   label = 'selected_feats', color='blue')

# plt.title('Density Accuracy Plot RF')
# plt.legend(prop={'size': 10})
# plt.show()

# sns.distplot(df['time_acc_gnb_original_feats'], hist = False, kde = True,
#                   kde_kws = {'shade': False, 'linewidth': 1}, 
#                   label = 'original_feats', color='red')

# sns.distplot(df['time_acc_gnb_selected_feats'], hist = False, kde = True,
#                   kde_kws = {'shade': False, 'linewidth': 1}, 
#                   label = 'selected_feats', color='blue')

# plt.title('Density Accuracy Plot GNB')
# plt.legend(prop={'size': 10})
# plt.show()

# sns.distplot(df['time_acc_dt_original_feats'], hist = False, kde = True,
#                   kde_kws = {'shade': False, 'linewidth': 1}, 
#                   label = 'original_feats', color='red')

# sns.distplot(df['time_acc_dt_selected_feats'], hist = False, kde = True,
#                   kde_kws = {'shade': False, 'linewidth': 1}, 
#                   label = 'selected_feats', color='blue')

# plt.title('Density Accuracy Plot dt')
# plt.legend(prop={'size': 10})
# plt.show()

# sns.distplot(df['time_acc_neight_original_feats'], hist = False, kde = True,
#                   kde_kws = {'shade': False, 'linewidth': 1}, 
#                   label = 'original_feats', color='red')

# sns.distplot(df['time_acc_neight_selected_feats'], hist = False, kde = True,
#                   kde_kws = {'shade': False, 'linewidth': 1}, 
#                   label = 'selected_feats', color='blue')

# plt.title('Density Accuracy Plot KNN')
# plt.legend(prop={'size': 10})
# plt.show()


# #### PRECISION ####

# sns.distplot(df['time_prec_rf_original_feats'], hist = False, kde = True,
#                   kde_kws = {'shade': False, 'linewidth': 1}, 
#                   label = 'original_feats', color='red')

# sns.distplot(df['time_prec_rf_selected_feats'], hist = False, kde = True,
#                   kde_kws = {'shade': False, 'linewidth': 1}, 
#                   label = 'selected_feats', color='blue')

# plt.title('Density Precision Plot RF')
# plt.legend(prop={'size': 10})
# plt.show()

# sns.distplot(df['time_prec_gnb_original_feats'], hist = False, kde = True,
#                   kde_kws = {'shade': False, 'linewidth': 1}, 
#                   label = 'original_feats', color='red')

# sns.distplot(df['time_prec_gnb_selected_feats'], hist = False, kde = True,
#                   kde_kws = {'shade': False, 'linewidth': 1}, 
#                   label = 'selected_feats', color='blue')

# plt.title('Density Precision Plot GNB')
# plt.legend(prop={'size': 10})
# plt.show()

# sns.distplot(df['time_prec_dt_original_feats'], hist = False, kde = True,
#                   kde_kws = {'shade': False, 'linewidth': 1}, 
#                   label = 'original_feats', color='red')

# sns.distplot(df['time_prec_dt_selected_feats'], hist = False, kde = True,
#                   kde_kws = {'shade': False, 'linewidth': 1}, 
#                   label = 'selected_feats', color='blue')

# plt.title('Density Precision Plot dt')
# plt.legend(prop={'size': 10})
# plt.show()

# sns.distplot(df['time_prec_neight_original_feats'], hist = False, kde = True,
#                   kde_kws = {'shade': False, 'linewidth': 1}, 
#                   label = 'original_feats', color='red')

# sns.distplot(df['time_prec_neight_selected_feats'], hist = False, kde = True,
#                   kde_kws = {'shade': False, 'linewidth': 1}, 
#                   label = 'selected_feats', color='blue')

# plt.title('Density Precision Plot KNN')
# plt.legend(prop={'size': 10})
# plt.show()


boxplot = df.boxplot(column=['time_acc_rf_gr', 'time_acc_rf_r', 'time_acc_rf_gr_r'])
plt.show()

boxplot = df.boxplot(column=['time_acc_gnb_gr', 'time_acc_gnb_r', 'time_acc_gnb_gr_r'])
plt.show()

boxplot = df.boxplot(column=['time_acc_mlp_gr', 'time_acc_mlp_r', 'time_acc_mlp_gr_r'])
plt.show()

boxplot = df.boxplot(column=['time_acc_svc_gr', 'time_acc_svc_r', 'time_acc_svc_gr_r'])
plt.show()

boxplot = df.boxplot(column=['time_acc_knn_gr', 'time_acc_knn_r', 'time_acc_knn_gr_r'])
plt.show()
# boxplot = df.boxplot(column=['time_acc_neight_original_feats', 'time_acc_neight_selected_feats'])
# plt.show()
# boxplot = df.boxplot(column=['time_acc_gnb_original_feats', 'time_acc_gnb_selected_feats'])
# plt.show()
# boxplot = df.boxplot(column=['time_acc_rf_original_feats', 'time_acc_rf_selected_feats'])
# plt.show()
# boxplot = df.boxplot(column=['time_acc_mlp_original_feats', 'time_acc_mlp_selected_feats'])
# plt.show()
# boxplot = df.boxplot(column=['time_acc_svc_original_feats', 'time_acc_svc_selected_feats'])
# plt.show()

# boxplot = df.boxplot(column=['time_prec_dt_original_feats', 'time_prec_dt_selected_feats'])
# plt.show()
# boxplot = df.boxplot(column=['time_prec_neight_original_feats', 'time_prec_neight_selected_feats'])
# plt.show()
# boxplot = df.boxplot(column=['time_prec_gnb_original_feats', 'time_prec_gnb_selected_feats'])
# plt.show()
# boxplot = df.boxplot(column=['time_prec_rf_original_feats', 'time_prec_rf_selected_feats'])
# plt.show()
# boxplot = df.boxplot(column=['time_prec_mlp_original_feats', 'time_prec_mlp_selected_feats'])
# plt.show()
# boxplot = df.boxplot(column=['time_prec_svc_original_feats', 'time_prec_svc_selected_feats'])
# plt.show()

# boxplot = df.boxplot(column=['time_rec_dt_original_feats', 'time_rec_dt_selected_feats'])
# plt.show()
# boxplot = df.boxplot(column=['time_rec_neight_original_feats', 'time_rec_neight_selected_feats'])
# plt.show()
# boxplot = df.boxplot(column=['time_rec_gnb_original_feats', 'time_rec_gnb_selected_feats'])
# plt.show()
# boxplot = df.boxplot(column=['time_rec_rf_original_feats', 'time_rec_rf_selected_feats'])
# plt.show()
# boxplot = df.boxplot(column=['time_rec_mlp_original_feats', 'time_rec_mlp_selected_feats'])
# plt.show()
# boxplot = df.boxplot(column=['time_rec_svc_original_feats', 'time_rec_svc_selected_feats'])
# plt.show()

# boxplot = df.boxplot(column=['time_f1_dt_original_feats', 'time_f1_dt_selected_feats'])
# plt.show()
# boxplot = df.boxplot(column=['time_f1_neight_original_feats', 'time_f1_neight_selected_feats'])
# plt.show()
# boxplot = df.boxplot(column=['time_f1_gnb_original_feats', 'time_f1_gnb_selected_feats'])
# plt.show()
# boxplot = df.boxplot(column=['time_f1_rf_original_feats', 'time_f1_rf_selected_feats'])
# plt.show()
# boxplot = df.boxplot(column=['time_f1_mlp_original_feats', 'time_f1_mlp_selected_feats'])
# plt.show()
# boxplot = df.boxplot(column=['time_f1_svc_original_feats', 'time_f1_svc_selected_feats'])
# plt.show()


# temp = df_selected_feats['age_at_diagnosis'].to_numpy()
# temp = pd.DataFrame({'age_at_diagnosis':temp})
# temp.hist()

# stat, p = shapiro(temp)
# print('statistics=%.6f, p=%.6f' % (stat, p))

# alpha = 0.05
# if p > alpha:
#  	print('Sample looks gaussian (fail to reject H0)')
# else:
#  	print('Sample does not look Gaussian (reject H0)')


