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


# df = pd.read_csv("metadata_1.csv", sep=';')
# df_temp = pd.read_csv("metadata_2.csv", sep=';')
df = pd.read_csv("metadata_3.csv", sep=';')

df.plot.density(figsize = (10, 10), linewidth=3)

# df = df.append(df_temp, ignore_index=True)

# temp = df['age_at_diagnosis'].to_numpy()
# # normalized_arr = preprocessing.normalize([temp])[0]
# temp = pd.DataFrame({'age_at_diagnosis':temp})
# # temp = temp[temp.age_at_diagnosis <= 1]

# temp.boxplot()
# qqplot(temp, line='s')
# plt.show()

# temp.hist(figsize=(100, 60))

# stat, p = shapiro(temp)
# print('statistics=%.3f, p=%.3f' % (stat, p))

# alpha = 0.05
# if p > alpha:
#  	print('sample looks gaussian (fail to reject H0)')
# else:
#  	print('Sample does not look Gaussian (reject H0)')

#### HISTOGRAMS

# df.hist(figsize=(100, 60))

#### BOXPLOT

plt.figure(figsize =(50, 30))

df.boxplot()

plt.xticks(rotation=90)

plt.show()

#### QQPLOT

# #time
# attribute = df["time"]
# qqplot(attribute, line='s')
# plt.title(label='time')
# plt.show()

# #radius_mean
# attribute = df["radius_mean"]
# qqplot(attribute, line='s')
# plt.title(label='radius_mean')
# plt.show()

# #texture_mean
# attribute = df["texture_mean"]
# qqplot(attribute, line='s')
# plt.title(label='texture_mean')
# plt.show()

# #perimeter_mean
# attribute = df["perimeter_mean"]
# qqplot(attribute, line='s')
# plt.title(label='perimeter_mean')
# plt.show()

# #area_mean
# attribute = df["area_mean"]
# qqplot(attribute, line='s')
# plt.title(label='area_mean')
# plt.show()

# #smoothness_mean
# attribute = df["smoothness_mean"]
# qqplot(attribute, line='s')
# plt.title(label='smoothness_mean')
# plt.show()

# #compactness_mean
# attribute = df["compactness_mean"]
# qqplot(attribute, line='s')
# plt.title(label='compactness_mean')
# plt.show()

# #concavity_mean
# attribute = df["concavity_mean"]
# qqplot(attribute, line='s')
# plt.title(label='concavity_mean')
# plt.show()

# #concave_points_mean
# attribute = df["concave_points_mean"]
# qqplot(attribute, line='s')
# plt.title(label='concave_points_mean')
# plt.show()

# #symmetry_mean
# attribute = df["symmetry_mean"]
# qqplot(attribute, line='s')
# plt.title(label='symmetry_mean')
# plt.show()

# #fractal_dimension_mean
# attribute = df["fractal_dimension_mean"]
# qqplot(attribute, line='s')
# plt.title(label='fractal_dimension_mean')
# plt.show()

# stat, p = shapiro(df['perimeter_mean'])
# print('Statistics=%.3f, p=%.3f' % (stat, p))

# alpha = 0.05
# if p > alpha:
# 	print('Sample looks Gaussian (fail to reject H0)')
# else:
# 	print('Sample does not look Gaussian (reject H0)')






