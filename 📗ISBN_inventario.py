# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:51:50 2024

@author: HPALACIOS
"""

import numpy as np
import pandas as pd
import matplotlib as pl
#import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import silhouette_score

import pandas as pd 
import numpy as np 
#import pickle 
import streamlit as st 
from PIL import Image 
import time


st.set_page_config(page_title="ISBN - inventario", page_icon="📈")

#st.markdown("# Plotting Demo")

st.sidebar.title("Analítica Avanzada")

######Insertar logo en Sidebar######################################################

def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

#my_logo = add_logo(logo_path="logo_SM.png", width=90, height=90)
#st.sidebar.image(my_logo)

#my_libro = add_logo(logo_path="open-book.jpg", width=60, height=30)
#st.sidebar.image(my_libro)

#st.sidebar.image(add_logo(logo_path="your/logo/path", width=50, height=60)) 

st.sidebar.header("📈 ISBN")

############Título ###############################
st.title("España - Análisis de ISBN") 


#################################################

# this is the main function in which we define our webpage  
def main(): 

    # the font and background color, the padding and the text to be displayed 
    html_temp = """ 
    <div style ="background-color:red;padding:5px"> 
    <h1 style ="color:white;text-align:center;">Stock de ISBN </h1> 
    </div> 
    """
    
    # this line allows us to display the front end aspects we have  
    # defined in the above code 
    st.markdown(html_temp, unsafe_allow_html = True) 

    result ="" 

main()

datos = pd.read_csv('t_ocr.csv', delimiter=',')

st.subheader('Dashboard de ISBN', divider=True)

st.subheader('Tabla datos de origen:', divider=False)
#datos

from datetime import datetime, timedelta
datos['timestamp'] = pd.to_datetime(datos['timestamp'])

#datos.dtypes

datos1 = pd.DataFrame(datos, columns=['timestamp','error'])

#datos1

#datos1.dtypes

##########################################################
import plotly.express as px
st.subheader('Tabla total Procesados:', divider=False) 
by_time = datos1.groupby((datos1["timestamp"])).count().reset_index()
#by_month.index = pd.PeriodIndex(by_month.index)
by_time

#by_month.describe()
#by_month.dtypes

fig = px.line(x=by_time['timestamp'], y=by_time['error'])
st.plotly_chart(fig)


#######################################################
st.subheader('Tabla Procesados correctamente:', divider=False) 
by_time2 = datos1.groupby((datos1["timestamp"])).sum().reset_index()
#by_month.index = pd.PeriodIndex(by_month.index)
by_time2

fig2 = px.line(x=by_time2['timestamp'], y=by_time2['error'])
st.plotly_chart(fig2)


st.subheader('Tabla Procesados vs Correctos:', divider=False) 
resultado = by_time.merge(by_time2, on="timestamp", how="left")

resultado.rename(columns={"error_x": "Procesados", "error_y": "Correctos"}, inplace=True)

resultado

fig3 = px.line(resultado, x="timestamp", y=resultado.columns,
              title='Procesados vs Corectos')
st.plotly_chart(fig3)
###############################################

st.subheader('Tasa de acierto:', divider=False) 

resultado['Tasa_acierto'] = resultado['Correctos']/resultado['Procesados']

acierto = pd.concat((resultado['timestamp'],resultado['Tasa_acierto']),axis = 1)

acierto

fig4 = px.line(x=acierto['timestamp'], y=acierto['Tasa_acierto'])
st.plotly_chart(fig4)

st.subheader('Tabla estadística:', divider=True) 

acierto_descr = acierto.describe()

acierto_descr['Tasa_acierto']

acierto = round(sum(resultado['Correctos'])*100/sum(resultado['Procesados']),3)

#acierto
st.success('La tasa de acierto es (%): {}'.format(acierto), icon="🚨")

############Procesados resample contar

datos1['timestamp'] = pd.to_datetime(datos1['timestamp'])

datos1.set_index('timestamp', inplace=True)

datos_resample = datos1.resample('D').count()

#datos_resample

############Procesados resample sumar

datos_resample2 = datos1.resample('D').sum()

#datos_resample2

#######################
st.subheader(' ', divider=True) 
st.subheader('Agrupación por día', divider=False) 

data_diaria = pd.merge(datos_resample, datos_resample2, left_index=True, right_index=True)

data_diaria.rename(columns={"error_x": "Procesados", "error_y": "Correctos"}, inplace=True)

data_diaria

fig5 = px.line(data_diaria)
st.plotly_chart(fig5)

####################################
def elimina_error(datos, valor):
#    datos
#    valor
    datos.drop(datos[(datos['error'] ==valor)].index, inplace=True)
    return datos


##########Resumen ########################
st.subheader(' ', divider=True) 
st.subheader('Resumen', divider=False) 

ISBN_proc = datos['isbn'].nunique()

st.success('Total de ISBN procesados: {}'.format(ISBN_proc))

#ISBN_proc

ISBN_val = elimina_error(datos, 0)

ISBN_validados = ISBN_val['isbn'].nunique()

st.success('Total de ISBN validados en el maestro: {}'.format(ISBN_validados))

Tasa_leidos = round(ISBN_validados*100/ISBN_proc,2)

st.success('Total de ISBN leídos válidos (%): {}'.format(Tasa_leidos))


#ISBN_validados


