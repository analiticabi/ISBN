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

import pandas as pd 
import numpy as np 
import pickle 
import streamlit as st 
from PIL import Image 
import time


st.set_page_config(page_title="Modelo de Cluster", page_icon="📈")

#st.markdown("# Plotting Demo")

st.sidebar.title("Analítica de Negocio")

######Insertar logo en Sidebar######################################################

def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

my_logo = add_logo(logo_path="C:/Users/hpalacios/Downloads/Logo_SM.png", width=90, height=90)
st.sidebar.image(my_logo)

#my_libro = add_logo(logo_path="C:/Users/hpalacios/Downloads/open-book.jpg", width=60, height=30)
#st.sidebar.image(my_libro)

#st.sidebar.image(add_logo(logo_path="your/logo/path", width=50, height=60)) 


st.sidebar.header("📈 Clustering Analysis")


# loading in the model to predict on the data 
pickle_in = open("C:/Users/hpalacios/Downloads/archive/clustering_km.pkl", 'rb') 
clustering_km = pickle.load(pickle_in) 

 
############Título ###############################
st.title("SM Chile - Clustering Analysis") 


#################################################

# this is the main function in which we define our webpage  
def main(): 

    # the font and background color, the padding and the text to be displayed 
    html_temp = """ 
    <div style ="background-color:red;padding:5px"> 
    <h1 style ="color:white;text-align:center;">Modelo de Clustering Colegios </h1> 
    </div> 
    """
    
    # this line allows us to display the front end aspects we have  
    # defined in the above code 
    st.markdown(html_temp, unsafe_allow_html = True) 

    result ="" 

main()


##########################################################################################

data = pd.read_excel('C:/Users/hpalacios/Downloads/Ventas_colegios.xlsx', usecols = ['Adopciones', 'Venta'])

#data = pd.read_excel('C:/Users/hpalacios/Downloads/Red_colegios.xlsx')

data = data.dropna()

st.subheader('Datos crudos:', divider=False)

data

data_sct_1 = data

st.scatter_chart(data_sct_1, x='Adopciones', y='Venta')


#############Dependencia variables heatmap

st.subheader('Heat Map:', divider=False)

import seaborn as sns
import plotly.express as px
import matplotlib as plt

plot = sns.heatmap(data.corr(), vmin=-1, vmax=+1, annot=True, cmap="coolwarm")

st.pyplot(plot.get_figure())

##################################################

st.subheader('Redefiniendo features para evitar colinealidad:', divider=False)
 #streamlit run C:/Users/hpalacios/Downloads/archive/Analitica_Principal.py

##############Definir "Tasa adopcion"######################################
data1 = pd.read_excel('C:/Users/hpalacios/Downloads/Ventas_colegios.xlsx')

data1 = data1.dropna()

data1['Tasa_adopcion'] = data1['Venta']/data1['Adopciones']

############### Se elijen otras dos variables

data1 = pd.concat((data1['Tasa_adopcion'],data1['Adop_unidad']),axis = 1)

data_sct_2 = data1

st.scatter_chart(data_sct_2, x='Tasa_adopcion',y='Adop_unidad')

plot_2 = sns.heatmap(data1.corr(), vmin=-1, vmax=+1, annot=True, cmap="coolwarm")

st.subheader('Heapmap con nuevos features:', divider=False)

st.pyplot(plot_2.get_figure())

################################################

#Se realiza el escalamiento de los datos
from sklearn import preprocessing

data_escalada = preprocessing.MinMaxScaler().fit_transform(data1)

#Se determina las variables a evaluar
X = data_escalada.copy()


W = pd.DataFrame(data = X, columns=['Tasa_adopcion','Adop_unidad'])

st.subheader('Datos escalados sin outlayers:', divider=False)

from K_M_def import elimina

Omega = elimina(W, 0.12)

Omega

st.scatter_chart(Omega, x='Tasa_adopcion', y='Adop_unidad')

#############################################################################
st.subheader('Análisis de N° de Clusters:', divider=False)

from K_M_def import codo_km
import streamlit as st


codo_km(W)


#############Volviendo a escalar datos y graficar

st.subheader('Conjunto de datos escalados:', divider=False)

data_escalada = preprocessing.MinMaxScaler().fit_transform(W)

#Se determina las variables a evaluar
X = data_escalada.copy()

W2 = pd.DataFrame(data = X, columns=['Tasa_adopcion','Adop_unidad'])

st.scatter_chart(W2, x='Tasa_adopcion',y='Adop_unidad')

#st.pyplot(plot.get_figure())

## Hallar el valor óptimo de K ##



#create your figure and get the figure object returned

import streamlit as st
import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components

#fig = plt.figure() 

N_Centros = 8

##################################

def segmentar(X, N_Centros):
        algoritmo = KMeans(n_clusters = N_Centros, init = 'k-means++', 
                           max_iter = 3000, n_init = 10)
        return algoritmo.fit(X)
    


#modelo_cluster()

st.subheader('# Clusters:', divider=False) 
      
     
    # the following lines create text boxes in which the user can enter  
    # the data required to make the prediction 
    
N_Centros = st.number_input("Elija Número de Clusters (2 < N° < 11)", min_value=3, max_value=10)


#print(segmentar(X, N_Centros))


## Se aplica el algoritmo de clustering ##
#Se define el algoritmo junto con el valor de K
algoritmo = KMeans(n_clusters = N_Centros, init = 'k-means++', 
                   max_iter = 3000, n_init = 10)

algoritmo.fit(Omega)


centroides, etiquetas = algoritmo.cluster_centers_, algoritmo.labels_
st.subheader('Centroides:', divider=False) 
#centroides

centroides2 = pd.DataFrame(centroides, columns =['Tasa_adopcion', 'Adop_unidad'])

centroides2

st.subheader('Gráfico de Centroides:', divider=False) 
#centroides

st.scatter_chart(
    centroides2,
    x='Tasa_adopcion',
    y='Adop_unidad',
    color='Adop_unidad',
#    size='Adop_unidad',
)


#st.subheader('Etiquetas:', divider=False) 
#etiquetas
##########################################

st.subheader('Cluster Datos originales:', divider=False) 

Z = pd.Series(data = etiquetas)


datares = data.dropna()


####Columna con Clusters
dataF = pd.DataFrame({'Cluster': Z})

data_fus = pd.merge(datares, dataF, left_index=True, right_index=True)
data_fus
st.subheader('Gráfico Cluster Datos originales:', divider=False) 

st.scatter_chart(
    data_fus,
    x='Adopciones',
    y='Venta',
    color='Cluster',
    size='Venta',
)


###################################
st.subheader('Cluster Conjunto de datos escalados:', divider=False) 
data_fus2 = pd.merge(W2, dataF, left_index=True, right_index=True)
data_fus2

st.subheader('Gráfico Cluster Conjunto de datos escalados:', divider=False) 

st.scatter_chart(
    data_fus2,
    x='Tasa_adopcion',
    y='Adop_unidad',
    color='Cluster',
#    size='Adop_unidad',
)


#data_escalada
st.subheader('Tabla Cluster por Colegio:', divider=False) 

data_cole = pd.read_excel('C:/Users/hpalacios/Downloads/Ventas_colegios.xlsx', usecols = ['Nombre _cuenta'])

#data = pd.read_excel('C:/Users/hpalacios/Downloads/Red_colegios.xlsx')

data_cole = data_cole.dropna()


data_fus3 = pd.merge(data_cole, data_fus2, left_index=True, right_index=True)
data_fus3