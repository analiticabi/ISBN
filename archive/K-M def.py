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

data = pd.read_excel('C:/Users/hpalacios/Downloads/Ventas_colegios.xlsx', usecols = ['Adopciones', 'Venta'])

#data = pd.read_excel('C:/Users/hpalacios/Downloads/Red_colegios.xlsx')

data = data.dropna()

data

plt.scatter(data['Adopciones'], data['Venta'],
            marker = 'x', s = 100, linewidths = 3)

#############Dependencia variables heatmap

import seaborn as sns

sns.heatmap(data.corr(), vmin=-1, vmax=+1, annot=True, cmap="coolwarm")

##############Definir "Tasa adopcion"######################################
data = pd.read_excel('C:/Users/hpalacios/Downloads/Ventas_colegios.xlsx')

data = data.dropna()

data

data['Tasa_adopcion'] = data['Venta']/data['Adopciones']

data

############### Se elijen otras dos variables

data = pd.concat((data['Tasa_adopcion'],data['Adop_unidad']),axis = 1)

data

plt.scatter(data['Tasa_adopcion'], data['Adop_unidad'],
            marker = 'x', s = 100, linewidths = 3)

## Se comprueba correlacion

sns.heatmap(data.corr(), vmin=-1, vmax=+1, annot=True, cmap="coolwarm")


################################################

#Se realiza el escalamiento de los datos
from sklearn import preprocessing

data_escalada = preprocessing.MinMaxScaler().fit_transform(data)

#Se determina las variables a evaluar
X = data_escalada.copy()


W = pd.DataFrame(data = X, columns=['Tasa_adopcion','Adop_unidad'])

W

plt.scatter(W['Tasa_adopcion'], W['Adop_unidad'],
            marker = 'x', s = 100, linewidths = 3)

###Eliminar outlyer
W.drop(W[(W['Tasa_adopcion'] >=1)].index, inplace=True)

W

plt.scatter(W['Tasa_adopcion'], W['Adop_unidad'],
            marker = 'x', s = 100, linewidths = 3)

W.drop(W[(W['Tasa_adopcion'] >=0.12)].index, inplace=True)

W

plt.scatter(W['Tasa_adopcion'], W['Adop_unidad'],
            marker = 'x', s = 100, linewidths = 3)



#############Volviendo a escalar datos y graficar

data_escalada = preprocessing.MinMaxScaler().fit_transform(W)

#Se determina las variables a evaluar
X = data_escalada.copy()

X  # NP

# A pandas

W = pd.DataFrame(data = X, columns=['Tasa_adopcion','Adop_unidad'])

W

plt.scatter(W['Tasa_adopcion'], W['Adop_unidad'],
            marker = 'x', s = 100, linewidths = 3)

## Hallar el valor óptimo de K ##

#Se aplicará el método de codo para hallar K

#Se calcula el algoritmo de agrupación para diferentes valores de K

def codo_km(Xa):

    inercia = [] 
    for i in range(1, 20):
        algoritmo = KMeans(n_clusters = i, init = 'k-means++', 
                       max_iter = 300, n_init = 10)
        algoritmo.fit(X)
    #Para cada K, se calcula la suma total del cuadrado dentro del clúster
        inercia.append(algoritmo.inertia_)

#Se traza la curva de la suma de errores cuadráticos 
    plt.figure(figsize=[10,6])
    plt.title('Método del Codo')
    plt.xlabel('No. de clusters')
    plt.ylabel('Inercia')
    plt.plot(list(range(1, 20)), inercia, marker='o')
    plt.show()
    return plt.plot(list(range(1, 20)), inercia, marker='o')
#    return plt.figure(figsize=[10,6]), plt.title('Método del Codo'), plt.xlabel('No. de clusters')
#    plt.ylabel('Inercia'), plt.plot(list(range(1, 20)), inercia, marker='o'), plt.show()

codo_km(W)

## Se aplica el algoritmo de clustering ##
#Se define el algoritmo junto con el valor de K
algoritmo = KMeans(n_clusters = 8, init = 'k-means++', 
                   max_iter = 3000, n_init = 10)

#########Se entrena el algoritmo

algoritmo.fit(X)

print(algoritmo.fit(X))


# pickling the model 
import pickle 
pickle_out = open("C:/Users/hpalacios/Downloads/archive/clustering_km.pkl", "wb") 
pickle.dump(algoritmo.fit, pickle_out) 
pickle_out.close()


import pickle 
pickle_out2 = open("C:/Users/hpalacios/Downloads/archive/centers_km.pkl", "wb") 
pickle.dump(algoritmo.cluster_centers_, pickle_out2) 
pickle_out.close()

#############Se obtiene los datos de los centroides y las etiquetas
centroides, etiquetas = algoritmo.cluster_centers_, algoritmo.labels_

etiquetas
centroides

data_escalada

Y = etiquetas.copy()

Y  # Etiquetas
X   #features

# Se define los colores de cada clúster
colores = ['blue', 'red', 'green', 'black', 'gray', 'orange', 'brown', 'yellow']

#Se asignan los colores a cada clústeres
colores_cluster = [colores[etiquetas[i]] for i in range(len(X))]

#Se grafica 
plt.scatter(X[:, 0], X[:, 1], c = colores_cluster, 
            marker = 'o',alpha = 0.4)

#Se grafican los centroides
plt.scatter(centroides[:, 0], centroides[:, 1],
            marker = 'x', s = 100, linewidths = 3, c = colores)


########### Unir data set con etiquetas########################################
## Tenemos etyiqueta y datos
Y        #en Numpy
data

Z = pd.Series(data = Y)
Z

datares = data.dropna()
datares

####Columna con Clusters
dataF = pd.DataFrame({'Cluster': Z})
dataF

#### Se fusionan 2 DF
data_fus = pd.merge(datares, dataF, left_index=True, right_index=True)
data_fus


############ Se transforma Np en DSataframe de pd
X

dataX = pd.DataFrame(X, columns = ['Tasa_adopcion','Adop_unidad'])
dataX

#######Se fusionan esos dos DF

dataX_fus = pd.merge(dataX, dataF, left_index=True, right_index=True)
dataX_fus


##############  Graficar en plotly

import plotly.express as px
import plotly.io as pio
pio.renderers.default = "svg"


Cl = px.scatter(dataX_fus, x="Tasa_adopcion", y="Adop_unidad",
                 color="Cluster", hover_name="Cluster",
                 log_x=False, size_max=60)


Cl.show()


###########Dejarlo disponible en Excel

from pandas import ExcelWriter

datos = dataX_fus.to_excel('C:/Users/hpalacios/Downloads/red_clus.xlsx')


###################### Opcional: PCA

### GRAFICAR LOS DATOS JUNTO A LOS RESULTADOS ###
# Se aplica la reducción de dimensionalidad a los datos
from sklearn.decomposition import PCA

modelo_pca = PCA(n_components = 2)
modelo_pca.fit(X)
pca = modelo_pca.transform(X) 

pca

#Se aplicar la reducción de dimsensionalidad a los centroides
centroides_pca = modelo_pca.transform(centroides)

# Se define los colores de cada clúster
colores = ['blue', 'red', 'green', 'black', 'gray', 'orange', 'brown']

#Se asignan los colores a cada clústeres
colores_cluster = [colores[etiquetas[i]] for i in range(len(pca))]

#Se grafica los componentes PCA
plt.scatter(pca[:, 0], pca[:, 1], c = colores_cluster, 
            marker = 'o',alpha = 0.4)

#Se grafican los centroides
plt.scatter(centroides_pca[:, 0], centroides_pca[:, 1],
            marker = 'x', s = 100, linewidths = 3, c = colores)

#Se guadan los datos en una variable para que sea fácil escribir el código
xvector = modelo_pca.components_[0] * max(pca[:,0])
yvector = modelo_pca.components_[1] * max(pca[:,1])
columnas = data.columns

#Se grafican los nombres de los clústeres con la distancia del vector
for i in range(len(columnas)):
    #Se grafican los vectores
    plt.arrow(0, 0, xvector[i], yvector[i], color = 'red', 
              width = 0.0005, head_width = 0.02, alpha = 0.75)
    #Se colocan los nombres
    plt.text(xvector[i], yvector[i], list(columnas)[i], color='blue', 
             alpha=0.75)

plt.show()



########### Fin ########################################

import numpy as np
import pandas as pd

my_array = np.array([[11, 22, 33], [44, 55, 66]])

my_array

df = pd.DataFrame(my_array, columns=['Column_A', 'Column_B', 'Column_C'])

print(df)
print(type(df))


def elimina_filas(df, valor):
    df
    valor
    df.drop(df[(df['Tasa_adopcion'] >=valor)].index, inplace=True)
    return df


def elimina(df, valor):
    df
    valor
    df.drop(df[(df['Tasa_adopcion'] >=valor)].index, inplace=True)
    return df

Zas = X

X

import pickle 
pickle_out = open("C:/Users/hpalacios/Downloads/archive/elimina_filas.pkl", "wb") 
pickle.dump(elimina_filas, pickle_out) 
pickle_out.close()

import pickle 
pickle_out3 = open("C:/Users/hpalacios/Downloads/archive/elimina.pkl", "wb") 
pickle.dump(elimina, pickle_out3) 
pickle_out.close()

borra_x = pickle.load(open("C:/Users/hpalacios/Downloads/archive/elimina_filas.pkl", "rb"))
# make predictions for test data


data_test = W

data_test

borra = borra_x(data_test,0.1)

borra