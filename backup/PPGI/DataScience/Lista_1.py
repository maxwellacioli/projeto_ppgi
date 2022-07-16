# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 17:47:01 2020

@author: Acioli
"""

# Questão 4
# Crie um conjunto de dados com duas variáveis V1 e V2, tal que: 
# a. Mediana de V1 < Média de V1 (0,5 ponto) 

import pandas as pd
import numpy as np

def question4a():
    while(True):
        mu, sigma = 0.5, 0.1 # mean and standard deviation
        v1 = np.random.normal(mu, sigma, 100)
        median_v1 = np.median(v1)
        mean_v1 = np.mean(v1)
        if(median_v1 < mean_v1):
            print("v1 median is: ",median_v1)
            print("v1 mean is: ", mean_v1)
            return v1
        
# b. Mediana de V2 > Média de V2 (0,5 ponto)         
def question4b():
    while(True):
        v2 = np.random.random_sample(100)
        median_v2 = np.median(v2)
        mean_v2 = np.mean(v2)
        if(median_v2 > mean_v2):
            print("v2 median is: ",median_v2)
            print("v2 mean is: ", mean_v2)
            return v2        
            
df = pd.DataFrame({'v1': question4a(), 'v2': question4b()})        

# Questão 5
#a. Mostra o histograma de cada variável

import matplotlib.pyplot as plt

def plotHistogram(data):
    plt.hist(data, 30)
    plt.grid(True)
    plt.show()

v1 = df['v1']
v2 = df['v2'] 
    
plotHistogram(v1)
plotHistogram(v2)

#b. Verifica se as variáveis seguem uma distribuição Normal 
#   (use teste de hipótese)

from scipy import stats

def checkNormalDistribution(data):
    stat , pvalue = stats.shapiro(data)
    # Formulação das hipóteses
    # H0 = A distribuição dos dados é normal
    # HA = A distribuição dos dados não é normal
    # Nivel de significância
    a = 0.05
    
    print("Pvalue = ", pvalue)
    print("SignificantLevel = ", a)
    
    if(pvalue < a):
        print('The distribution is not normal')
    else:
        print('The distribution is normal')
        
    
checkNormalDistribution(v1)
checkNormalDistribution(v2)

# Questão 10

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

iris = load_iris()

X = iris.data
y = iris.target

kf = KFold(n_splits=10)

neigh = KNeighborsClassifier(n_neighbors=3)

f_measure_score = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    neigh.fit(X_train, y_train)
    pred = neigh.predict(X_test)
    f_measure_score.append(f1_score(y_test, pred, average='weighted'))
    
    
print(np.mean(f_measure_score))

    







