# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from statsmodels.graphics.gofplots import qqplot
from sklearn import preprocessing
import seaborn as sns


df = pd.read_csv("metrabric_rna_mutation.csv")

#Remove linhas cujo lymph_node_status não foi computado
# df = df[df.lymph_node_status != "?"]

df = df.drop(['patient_id'], axis=1)


columns_names = list(df)

df.dropna(inplace=True)

df.reset_index(drop=True, inplace=True)

index_cancer_type_detailed_to_remove = []

for index, row in df.iterrows():
    value = row['cancer_type_detailed'].strip()
    
    if(value == 'Breast'):
        index_cancer_type_detailed_to_remove.append(index) 
           
df.drop(df.index[index_cancer_type_detailed_to_remove], inplace=True )
           



# Tipo de cirurgia realizada
df = df.replace('MASTECTOMY', 1)
df = df.replace('BREAST CONSERVING', 2)

# Tipo de cancer
df = df.replace('Breast Cancer', 1)
df = df.replace('Breast Sarcoma', 2)

# Detalhe do Tipo de Cancer
df = df.replace('Breast Invasive Ductal Carcinoma', 1)
df = df.replace('Breast Mixed Ductal and Lobular Carcinoma', 2)
df = df.replace('Breast Invasive Lobular Carcinoma', 3)
df = df.replace('Breast Invasive Mixed Mucinous Carcinoma', 4)
df = df.replace('Metaplastic Breast Cancer', 5)

# Cellularity
df = df.replace('Low', 1)
df = df.replace('Moderate', 2)
df = df.replace('High', 3)

# Positive/Negative
df = df.replace('Negative', 0)
df = df.replace('Positive', 1)
df = df.replace('Positve', 1)

# her2statusmeasuredbysnp6
df = df.replace('LOSS', 1)
df = df.replace('NEUTRAL', 2)
df = df.replace('GAIN', 3)

df.to_csv('metabric_treated_data.csv', sep=',', encoding='utf-8', index=False)

age_at_diagnosis_mastectomy = df.loc[df['type_of_breast_surgery'] == 1, 
                                     'age_at_diagnosis']
age_at_diagnosis_breast_conserving = df.loc[df['type_of_breast_surgery'] == 2, 
                                            'age_at_diagnosis']


sns.distplot(a=age_at_diagnosis_mastectomy, bins=40, color='green',
             hist_kws={"edgecolor": 'grey'})

sns.distplot(a=age_at_diagnosis_breast_conserving, bins=40, color='blue',
             hist_kws={"edgecolor": 'grey'})

plt.legend(labels=
           ['Mastectomy', 'Breast Conserving'])
plt.show()



pik3ca_mut = df['pik3ca_mut'].unique()



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
#  	print('sample looks gaussian (fail to reject h0)')
# else:
#  	print('Sample does not look Gaussian (reject H0)')

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






