# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:22:58 2024

@author: HPALACIOS
"""


import pandas as pd 
import numpy as np 
import pickle 
import streamlit as st 
from PIL import Image 
import time


# loading in the model to predict on the data 
pickle_in = open("C:/Users/hpalacios/Downloads/archive/forecast.pkl", 'rb') 
forecast = pickle.load(pickle_in) 

#########Configurar etiqueta de la página web#################

st.set_page_config(page_title="Modelo Serie de Tiempo", page_icon="📈")

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


st.sidebar.header("📈 Serie de Tiempo")


############Título ###############################
st.title("SM Chile - Serie de Tiempo") 


#################################################

# this is the main function in which we define our webpage  
def main(): 

    # the font and background color, the padding and the text to be displayed 
    html_temp = """ 
    <div style ="background-color:red;padding:5px"> 
    <h1 style ="color:white;text-align:center;">Modelo de Forecast Ventas Online </h1> 
    </div> 
    """
    
    # this line allows us to display the front end aspects we have  
    # defined in the above code 
    st.markdown(html_temp, unsafe_allow_html = True) 

    result ="" 

main()

datos = pd.read_csv('C:/Users/hpalacios/Downloads/Ventas_diarias_SM_depurado.csv', delimiter=';')

st.subheader('Datos de Ventas Online:', divider=False)
datos

X_all = datos[['Tasa_desempleo','Ingles_digital',	'Ingles_impreso', 'LIJ_digital', 'LIJ_impreso', 'Religion_digital','Religion_impreso', 'Texto_complementario_digital', 'Texto_complementario_impreso', 'Texto_curricular_digital',	'Texto_curricular_impreso',
'year','month','day',
       'day_week','week_number']]

X_hm = datos[['Tasa_desempleo','Ingles_digital',	'Ingles_impreso', 'LIJ_digital', 'LIJ_impreso', 'Religion_digital','Religion_impreso', 'Texto_complementario_digital', 'Texto_complementario_impreso', 'Texto_curricular_digital',	'Texto_curricular_impreso',
'year','month','day',
       'day_week','week_number','monto']]

####################
st.subheader('Heat Map:', divider=False)
import seaborn as sns
import plotly.express as px
import matplotlib as plt


import seaborn as sns
#df_aux = df_aux.drop(columns='result')
plot = sns.heatmap(X_hm.corr(), vmin=-1, vmax=+1, annot=True, cmap="coolwarm")



st.pyplot(plot.get_figure())

########Predicción sobre el Dataset#######################################################

datos['monto_Prediction'] = forecast(X_all)

#########Plot de comparación##################################################

st.subheader('Serie de Tiempo:', divider=False)

#serie = datos[['monto','monto_Prediction']].plot(figsize=(15, 5))
st.line_chart(datos[['monto','monto_Prediction']])

###############################################

st.subheader('Forecast anual:', divider=False)


datos_extrap = pd.read_csv('C:/Users/hpalacios/Downloads/Ventas_diarias_SM_depurado_extrap.csv', delimiter=';')



X_all_extrap = datos_extrap[['Tasa_desempleo','Ingles_digital', 'Ingles_impreso', 'LIJ_digital', 'LIJ_impreso', 'Religion_digital','Religion_impreso', 'Texto_complementario_digital', 'Texto_complementario_impreso', 'Texto_curricular_digital',	'Texto_curricular_impreso',
'year','month','day',
       'day_week','week_number']]

X_hm_extrap = datos_extrap[['Tasa_desempleo','Ingles_digital', 'Ingles_impreso', 'LIJ_digital', 'LIJ_impreso', 'Religion_digital','Religion_impreso', 'Texto_complementario_digital', 'Texto_complementario_impreso', 'Texto_curricular_digital',	'Texto_curricular_impreso',
'year','month','day',
       'day_week','week_number','monto']]


datos_extrap['monto_Prediction'] = forecast(X_all_extrap)

#########Plot de comparación y extrapolación##################################################


#serie = datos[['monto','monto_Prediction']].plot(figsize=(15, 5))
st.line_chart(datos_extrap[['monto','monto_Prediction']])


#st.bar_chart(datos[['monto','monto_Prediction']])

#############################################

datos['residuo'] = abs(datos['monto_Prediction']-datos['monto'])

error_residuo = sum(datos['residuo'])*100/sum(datos['monto'])


 
st.subheader('Error(%):', divider=False)

error_residuo

st.subheader('Datos con Forecast:', divider=False)

datos

#############################################

     
#if __name__=='__main__': 
#    main()