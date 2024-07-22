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
pickle_in = open("C:/Users/hpalacios/Downloads/archive/classifierxgb.pkl", 'rb') 
classifier = pickle.load(pickle_in) 

#########Configurar etiqueta de la página web#################

st.set_page_config(page_title="Modelo XGB Demo", page_icon="📈")

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


st.sidebar.header("📈 XGB Demo")


#################################################


def prediction(Cursov, Dias_Restantesv, Paginasv, Partesv, Dias_cortev, Dias_sin_firmarv, Dias_transcurridos_sin_pedidov, Dias_sin_ingresar_almacenv): 
    datosv = [[Cursov,	Dias_Restantesv, Paginasv, Partesv, Dias_cortev, Dias_sin_firmarv, Dias_transcurridos_sin_pedidov, Dias_sin_ingresar_almacenv]]
    Curso = 'Curso'
    Dias_Restantes= 'Dias Restantes'
    Paginas = 'Paginas'
    Partes = 'Partes'
    Dias_corte = 'Dias corte'
    Dias_sin_firmar = 'Dias sin firmar'
    Dias_transcurridos_sin_pedido = 'Dias transcurridos sin pedido'
    Dias_sin_ingresar_almacen = 'Dias sin ingresar a almacen'
    columnas = [[Curso,	Dias_Restantes, Paginas, Partes, Dias_corte, Dias_sin_firmar, Dias_transcurridos_sin_pedido, Dias_sin_ingresar_almacen]]
    df = pd.DataFrame(datosv, columns=columnas)
    prediction = classifier.predict(df)
    print(prediction)
    return prediction 

################################

	
# this is the main function in which we define our webpage  
def main(): 
      # giving the webpage a title 
    st.title("SM Alerta de producción de Libros") 
      
    # here we define some of the front end elements of the web page like  
    # the font and background color, the padding and the text to be displayed 
    html_temp = """ 
    <div style ="background-color:red;padding:5px"> 
    <h1 style ="color:white;text-align:center;">Formulario - Modelo Clasificador </h1> 
    </div> 
    """
    
    # this line allows us to display the front end aspects we have  
    # defined in the above code 
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # the following lines create text boxes in which the user can enter  
    # the data required to make the prediction 
    
    Cursoz = st.number_input("Curso [Rango: 30-36]", min_value=30, max_value=36)
    Dias_Restantesz= st.number_input("Días Restantes")
    Paginasz = st.number_input("Número de Páginas",min_value=0, max_value=10000)
    Partesz = st.number_input("Cantidad de Partes",min_value=0, max_value=10000)
    Dias_cortez = st.number_input("Días de corte")
    Dias_sin_firmarz = st.number_input("Días sin firmar")
    Dias_transcurridos_sin_pedidoz = st.number_input("Días transcurridos sin pedido")
    Dias_sin_ingresar_almacenz = st.number_input("Días sin ingresar en almacén") 
    result ="" 
      
    # the below line ensures that when the button called 'Predict' is clicked,  
    # the prediction function defined above is called to make the prediction  
    # and store it in the variable result 
#    if st.button("Clasificar"): 
#        result = prediction(Curso, Dias_Restantes, Paginas, Partes, Dias_corte, Dias_sin_firmar, Dias_transcurridos_sin_pedido, Dias_sin_ingresar_almacen) 
#        st.success('El resultado es {}'.format(result)) 
#        if result == 1: 
#            st.title("En tiempo")
#        else:
#            st.title("Alerta")
    if st.button("Clasificar"): 
        result = prediction(Cursov = Cursoz, Dias_Restantesv= Dias_Restantesz, Paginasv = Paginasz, Partesv = Partesz, Dias_cortev = Dias_cortez, Dias_sin_firmarv = Dias_sin_firmarz, Dias_transcurridos_sin_pedidov = Dias_transcurridos_sin_pedidoz, Dias_sin_ingresar_almacenv = Dias_sin_ingresar_almacenz) 
        if result == 1: st.success('En tiempo {}'.format(result), icon="✅") 
        else: st.success('Alerta {}'.format(result),icon="🚨") 
     
if __name__=='__main__': 
    main()