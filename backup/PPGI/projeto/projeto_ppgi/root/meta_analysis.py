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


df_0 = pd.read_csv("metadata1.csv", sep=';')
df_1 = pd.read_csv("metadata2.csv", sep=';')
df = pd.read_csv("metadata3.csv", sep=';')

df_selected_feats = pd.read_csv("selected_features_data.csv", sep=',')
columns_names_selected_feats = list(df_selected_feats)

#### HISTOGRAMS

# df.hist(figsize=(100, 60))

#### NEW DATA ####

df = df.append(df_0, ignore_index=True)
df = df.append(df_1, ignore_index=True)

columns_names = list(df)


#### ACCURACY ####

sns.distplot(df['time_acc_rf_original_feats'], hist = False, kde = True,
                  kde_kws = {'shade': False, 'linewidth': 1}, 
                  label = 'original_feats', color='red')

sns.distplot(df['time_acc_rf_selected_feats'], hist = False, kde = True,
                  kde_kws = {'shade': False, 'linewidth': 1}, 
                  label = 'selected_feats', color='blue')

plt.title('Density Accuracy Plot RF')
plt.legend(prop={'size': 10})
plt.show()

sns.distplot(df['time_acc_gnb_original_feats'], hist = False, kde = True,
                  kde_kws = {'shade': False, 'linewidth': 1}, 
                  label = 'original_feats', color='red')

sns.distplot(df['time_acc_gnb_selected_feats'], hist = False, kde = True,
                  kde_kws = {'shade': False, 'linewidth': 1}, 
                  label = 'selected_feats', color='blue')

plt.title('Density Accuracy Plot GNB')
plt.legend(prop={'size': 10})
plt.show()

sns.distplot(df['time_acc_lr_original_feats'], hist = False, kde = True,
                  kde_kws = {'shade': False, 'linewidth': 1}, 
                  label = 'original_feats', color='red')

sns.distplot(df['time_acc_lr_selected_feats'], hist = False, kde = True,
                  kde_kws = {'shade': False, 'linewidth': 1}, 
                  label = 'selected_feats', color='blue')

plt.title('Density Accuracy Plot LR')
plt.legend(prop={'size': 10})
plt.show()

sns.distplot(df['time_acc_neight_original_feats'], hist = False, kde = True,
                  kde_kws = {'shade': False, 'linewidth': 1}, 
                  label = 'original_feats', color='red')

sns.distplot(df['time_acc_neight_selected_feats'], hist = False, kde = True,
                  kde_kws = {'shade': False, 'linewidth': 1}, 
                  label = 'selected_feats', color='blue')

plt.title('Density Accuracy Plot KNN')
plt.legend(prop={'size': 10})
plt.show()


#### PRECISION ####

sns.distplot(df['time_prec_rf_original_feats'], hist = False, kde = True,
                  kde_kws = {'shade': False, 'linewidth': 1}, 
                  label = 'original_feats', color='red')

sns.distplot(df['time_prec_rf_selected_feats'], hist = False, kde = True,
                  kde_kws = {'shade': False, 'linewidth': 1}, 
                  label = 'selected_feats', color='blue')

plt.title('Density Precision Plot RF')
plt.legend(prop={'size': 10})
plt.show()

sns.distplot(df['time_prec_gnb_original_feats'], hist = False, kde = True,
                  kde_kws = {'shade': False, 'linewidth': 1}, 
                  label = 'original_feats', color='red')

sns.distplot(df['time_prec_gnb_selected_feats'], hist = False, kde = True,
                  kde_kws = {'shade': False, 'linewidth': 1}, 
                  label = 'selected_feats', color='blue')

plt.title('Density Precision Plot GNB')
plt.legend(prop={'size': 10})
plt.show()

sns.distplot(df['time_prec_lr_original_feats'], hist = False, kde = True,
                  kde_kws = {'shade': False, 'linewidth': 1}, 
                  label = 'original_feats', color='red')

sns.distplot(df['time_prec_lr_selected_feats'], hist = False, kde = True,
                  kde_kws = {'shade': False, 'linewidth': 1}, 
                  label = 'selected_feats', color='blue')

plt.title('Density Precision Plot LR')
plt.legend(prop={'size': 10})
plt.show()

sns.distplot(df['time_prec_neight_original_feats'], hist = False, kde = True,
                  kde_kws = {'shade': False, 'linewidth': 1}, 
                  label = 'original_feats', color='red')

sns.distplot(df['time_prec_neight_selected_feats'], hist = False, kde = True,
                  kde_kws = {'shade': False, 'linewidth': 1}, 
                  label = 'selected_feats', color='blue')

plt.title('Density Precision Plot KNN')
plt.legend(prop={'size': 10})
plt.show()


boxplot = df.boxplot(column=['time_acc_lr_original_feats', 'time_acc_lr_selected_feats'])
plt.show()
boxplot = df.boxplot(column=['time_acc_neight_original_feats', 'time_acc_neight_selected_feats'])
plt.show()
boxplot = df.boxplot(column=['time_acc_gnb_original_feats', 'time_acc_gnb_selected_feats'])
plt.show()
boxplot = df.boxplot(column=['time_acc_rf_original_feats', 'time_acc_rf_selected_feats'])
plt.show()

boxplot = df.boxplot(column=['time_prec_lr_original_feats', 'time_prec_lr_selected_feats'])
plt.show()
boxplot = df.boxplot(column=['time_prec_neight_original_feats', 'time_prec_neight_selected_feats'])
plt.show()
boxplot = df.boxplot(column=['time_prec_gnb_original_feats', 'time_prec_gnb_selected_feats'])
plt.show()
boxplot = df.boxplot(column=['time_prec_rf_original_feats', 'time_prec_rf_selected_feats'])
plt.show()

boxplot = df.boxplot(column=['time_rec_lr_original_feats', 'time_rec_lr_selected_feats'])
plt.show()
boxplot = df.boxplot(column=['time_rec_neight_original_feats', 'time_rec_neight_selected_feats'])
plt.show()
boxplot = df.boxplot(column=['time_rec_gnb_original_feats', 'time_rec_gnb_selected_feats'])
plt.show()
boxplot = df.boxplot(column=['time_rec_rf_original_feats', 'time_rec_rf_selected_feats'])
plt.show()

boxplot = df.boxplot(column=['time_f1_lr_original_feats', 'time_f1_lr_selected_feats'])
plt.show()
boxplot = df.boxplot(column=['time_f1_neight_original_feats', 'time_f1_neight_selected_feats'])
plt.show()
boxplot = df.boxplot(column=['time_f1_gnb_original_feats', 'time_f1_gnb_selected_feats'])
plt.show()
boxplot = df.boxplot(column=['time_f1_rf_original_feats', 'time_f1_rf_selected_feats'])
plt.show()


temp = df_selected_feats['age_at_diagnosis'].to_numpy()
temp = pd.DataFrame({'age_at_diagnosis':temp})
temp.hist()

stat, p = shapiro(temp)
print('statistics=%.6f, p=%.6f' % (stat, p))

alpha = 0.05
if p > alpha:
 	print('Sample looks gaussian (fail to reject H0)')
else:
 	print('Sample does not look Gaussian (reject H0)')


