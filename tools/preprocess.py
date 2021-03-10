# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 08:41:30 2020

@author: Acioli
"""


import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess(file_name='../tools/wpbc.csv'):

    #load dataset
    patients = pd.read_csv(file_name)

    # remove id column
    patients = patients.drop(['id'], axis=1)
    
    #exclusão das instancia cujo fator não foi informado
    patients = patients[patients['lymph_node_status'] != '?']
    
    patients = patients.replace('R', 1)
    patients = patients.replace('N', 0)

    # from pandas_profiling import ProfileReport
    # profile = ProfileReport(patients, title='Explanatory Analisys', 
    #                         html={'style':{'full_width':True}})
    # profile.to_notebook_iframe()
    # profile.to_file(output_file="dataframe_report.html")

    #mover coluna situação_atual para o final do dataframe
    target = patients.pop('outcome')

    # print(patients.info())

    # # #Dividir o conjunto de informações em dados de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(patients, target, 
                                                        test_size=0.2, 
                                                        random_state=42)
    
    return X_train, X_test, y_train, y_test, patients, target

