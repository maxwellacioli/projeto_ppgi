# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from statsmodels.graphics.gofplots import qqplot
from sklearn import preprocessing


df = pd.read_csv("recurrence_data_set.csv")

#Remove linhas cujo lymph_node_status não foi computado
df = df[df.lymph_node_status != "?"]

# df = df.drop(['id_number', 'outcome', 'lymph_node_status'], axis=1)

temp = df['area_mean'].to_numpy()
normalized_arr = preprocessing.normalize([temp])[0]
temp = pd.DataFrame({'area_mean':normalized_arr})
# temp = temp[temp.area_mean <= 1]

temp.boxplot()
qqplot(temp, line='s')
plt.show()

temp.hist(figsize=(100, 60))

stat, p = shapiro(temp)
print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha = 0.05
if p > alpha:
 	print('Sample looks Gaussian (fail to reject H0)')
else:
 	print('Sample does not look Gaussian (reject H0)')

#### HISTOGRAMS

# df.hist(figsize=(100, 60))

#### BOXPLOT

# plt.figure(figsize =(50, 30))

# df.boxplot()

# plt.xticks(rotation=90)

# plt.show()

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





