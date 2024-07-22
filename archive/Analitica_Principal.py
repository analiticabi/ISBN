# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:17:28 2024

@author: HPALACIOS
"""

import streamlit as st
from PIL import Image 

st.set_page_config(page_title="Herramientas Analiticas", page_icon="📈")

st.sidebar.title("Analítica de Negocio")

def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

my_logo = add_logo(logo_path="C:/Users/hpalacios/Downloads/Logo_SM.png", width=90, height=90)
st.sidebar.image(my_logo)


my_logo2 = add_logo(logo_path="C:/Users/hpalacios/Downloads/Logo_SM.png", width=50, height=50)
image = Image.open("C:/Users/hpalacios/Downloads/open-book.jpg")
st.image(my_logo2)

st.title("Portal de Herramientas Analíticas") 
html_temp = """ 
    <div style ="background-color:red;padding:5px"> 
    <h1 style ="color:white;text-align:center;">Elegir herramienta en Menú lateral</h1> 
    </div> 
    </div> 
    </div> 
    <div style ="background-color:white;padding:1px"> 
    <h1 style ="color:white;text-align:center;"> </h1> 
    """
    
    # this line allows us to display the front end aspects we have  
    # defined in the above code 
st.markdown(html_temp, unsafe_allow_html = True) 
      
st.sidebar.markdown("    ")

def main_page():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")

def page2():
    st.markdown("# Page 2 ❄️")
    st.sidebar.markdown("# Page 2 ❄️")

def page3():
    st.markdown("# Page 3 🎉")
    st.sidebar.markdown("# Page 3 🎉")
    
my_libro2 = add_logo(logo_path="C:/Users/hpalacios/Downloads/open-book.jpg", width=100, height=40)
image = Image.open("C:/Users/hpalacios/Downloads/open-book.jpg")
st.image(my_libro2,'Equipo Analítica de Negocio')

#page_names_to_funcs = {
#    "Main Page": main_page,
#    "Page 2": page2,
#    "Page 3": page3,
#}

#selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
#page_names_to_funcs[selected_page]()