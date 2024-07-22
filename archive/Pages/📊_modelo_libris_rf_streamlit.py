# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:42:54 2024

@author: HPALACIOS
"""


import pandas as pd 
import numpy as np 
import pickle 
import streamlit as st 
from PIL import Image 
import time


# loading in the model to predict on the data 
pickle_in = open("C:/Users/hpalacios/Downloads/archive/classifier.pkl", 'rb') 
classifier = pickle.load(pickle_in) 

#########Configurar etiqueta de la página web#################

st.set_page_config(page_title="Modelo Random Forest Demo", page_icon="📈")

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


st.sidebar.header("📈 Random Forest Demo")


#################################################


def prediction(Curso, Dias_Restantes, Paginas, Partes, Dias_corte, Dias_sin_firmar, Dias_transcurridos_sin_pedido, Dias_sin_ingresar_almacen): 
    prediction = classifier.predict([[Curso,	Dias_Restantes, Paginas, Partes, Dias_corte, Dias_sin_firmar, Dias_transcurridos_sin_pedido, Dias_sin_ingresar_almacen]])
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
    
    Curso = st.number_input("Curso [Rango: 30-36]", min_value=30, max_value=36)
    Dias_Restantes= st.text_input("Días Restantes")
    Paginas = st.number_input("Número de Páginas",min_value=0, max_value=10000)
    Partes = st.number_input("Cantidad de Partes",min_value=0, max_value=10000)
    Dias_corte = st.text_input("Días de corte")
    Dias_sin_firmar = st.text_input("Días sin firmar")
    Dias_transcurridos_sin_pedido = st.text_input("Días transcurridos sin pedido")
    Dias_sin_ingresar_almacen = st.text_input("Días sin ingresar en almacén") 
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
        result = prediction(Curso, Dias_Restantes, Paginas, Partes, Dias_corte, Dias_sin_firmar, Dias_transcurridos_sin_pedido, Dias_sin_ingresar_almacen) 
        if result == 1: st.success('En tiempo {}'.format(result), icon="✅") 
        else: st.success('Alerta {}'.format(result),icon="🚨") 
     
if __name__=='__main__': 
    main()