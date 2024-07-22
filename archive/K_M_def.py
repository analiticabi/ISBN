# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:59:08 2023

@author: HPALACIOS
"""

import numpy as np
import pandas as pd
#import matplotlib as pl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import streamlit as st
import seaborn as sns

def codo_km(Xa):

    inercia = [] 
    for i in range(1, 20):
        algoritmo = KMeans(n_clusters = i, init = 'k-means++', 
                       max_iter = 300, n_init = 10)
        algoritmo.fit(Xa)
    #Para cada K, se calcula la suma total del cuadrado dentro del clúster
        inercia.append(algoritmo.inertia_)

#Se traza la curva de la suma de errores cuadráticos 
    plt.figure(figsize=[10,6])
    plt.title('Método del Codo')
    plt.xlabel('No. de clusters')
    plt.ylabel('Inercia')
    plt.plot(list(range(1, 20)), inercia, marker='o')
    #plt.show()
    st.pyplot(plt.gcf())
    return plt.plot(list(range(1, 20)), inercia, marker='o')
#    return plt.figure(figsize=[10,6]), plt.title('Método del Codo'), plt.xlabel('No. de clusters')
#    plt.ylabel('Inercia'), plt.plot(list(range(1, 20)), inercia, marker='o'), plt.show()

# pickling the model 
import pickle 
pickle_out = open("C:/Users/hpalacios/Downloads/archive/elbow_km.pkl", "wb") 
pickle.dump(codo_km, pickle_out) 
pickle_out.close()


####################Graficar clusters


def grafico_cluster(X, centroides, etiquetas):

# Se define los colores de cada clúster
    colores = ['blue', 'red', 'green', 'black', 'gray', 'orange', 'brown']
#Se asignan los colores a cada clústeres
    colores_cluster = [colores[etiquetas[i]] for i in range(len(X))]

#Se grafica 
    plt.scatter(X[:, 0], X[:, 1], c = colores_cluster, 
            marker = 'o',alpha = 0.4)

#Se grafican los centroides

    return plt.scatter(centroides[:, 0], centroides[:, 1],
            marker = 'x', s = 100, linewidths = 3, c = colores)

# pickling the model 
import pickle 
pickle_out5 = open("C:/Users/hpalacios/Downloads/archive/grafico_cluster.pkl", "wb") 
pickle.dump(grafico_cluster, pickle_out5) 
pickle_out5.close()


def elimina(df, valor):
    df
    valor
    df.drop(df[(df['Tasa_adopcion'] >=valor)].index, inplace=True)
    return df

