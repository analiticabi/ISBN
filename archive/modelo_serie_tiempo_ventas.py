# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:39:30 2024

@author: HPALACIOS
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
plt.style.use('fivethirtyeight')

#######################################################

# load data
datos = pd.read_csv('C:/Users/hpalacios/Downloads/Ventas_diarias_SM_depurado.csv', delimiter=';')

datos

y= datos['monto']
y

#datos['fecha'] = pd.to_datetime(datos['fecha'], format="%Y/%m/%d, %H:%M:%S")
#
x= datos['fecha']
x

##########Plot###########################################

plt.plot_date(x, y)

plt.gcf().set_size_inches(9, 7)
plt.show()

datos.plot(style='.', figsize=(15,5), title='Ventas')
####Plot embellecido#####################################

color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]
datos.plot(style='.', figsize=(15,5), color=color_pal[0], title='Ventas')

##############################################################

split_date = 2024
datos_train = datos.loc[datos.year < split_date].copy()
datos_test = datos.loc[datos.year >= split_date].copy()

datos_train
datos_test

################Saltar###############################################

datos_test \
    .rename(columns={'monto': 'TEST SET'}) \
    .join(datos_train.rename(columns={'monto': 'TRAINING SET'}), how='outer') \
    .plot(figsize=(15,5), title='Ventas', style='.')

################################################################

def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    X = df[['dayofweek','quarter','month','year',
       'dayofyear','dayofmonth']]
    if label:
        y = df[label]
        return X, y
    return X

########################################################

create_features(datos_train, label='Ventas train')

X_train, y_train = create_features(datos_train, label='monto')
X_test, y_test = create_features(datos_test, label='monto')

datos['fecha'].dt.dayofweek

datos['dayofweek'] = datos['fecha'].dt.dayofweek
datos['quarter'] = datos['fecha'].dt.quarter
datos['month'] = datos['fecha'].dt.month
datos['year'] = datos['fecha'].dt.year
datos['dayofyear'] = datos['fecha'].dt.dayofyear
datos['dayofmonth'] = datos['fecha'].dt.day
#datos['weekofyear'] = datos['fecha'].dt.weekofyear

datosx = datos[['fecha', 'monto','dayofweek','quarter','month','year',
       'dayofyear','dayofmonth']]

datosx

#####################################################################

split_date = '2023-01-01'
datosx_train = datosx.loc[datosx.fecha < split_date].copy()
datosx_test = datosx.loc[datosx.fecha >= split_date].copy()

datosx_train
datosx_test

############Retomar

X_train = datos_train[['Tasa_desempleo','Ingles_digital',	'Ingles_impreso', 'LIJ_digital', 'LIJ_impreso', 'Religion_digital','Religion_impreso', 'Texto_complementario_digital', 'Texto_complementario_impreso', 'Texto_curricular_digital',	'Texto_curricular_impreso',
'year','month','day',
       'day_week','week_number']]

X_train
y_train = datos_train['monto']
y_train

X_test = datos_test[['Tasa_desempleo','Ingles_digital', 'Ingles_impreso', 'LIJ_digital', 'LIJ_impreso', 'Religion_digital','Religion_impreso', 'Texto_complementario_digital', 'Texto_complementario_impreso', 'Texto_curricular_digital',	'Texto_curricular_impreso',
'year','month','day',
       'day_week','week_number']]

X_test
y_test = datos_test['monto']
y_test



######Create XGBoost Model################################################

reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
       verbose=False) # Change verbose to True if you want to see it train

###########Plot importance###########################################

plot_importance(reg, height=0.9)

############################################################

datos_test['monto_Prediction'] = reg.predict(X_test)

datos_test

datos_all = pd.concat([datos_test, datos_train], sort=False)

datos_all


#######Sobre todo el conjunto##########################################################

X_all = datos[['Tasa_desempleo','Ingles_digital',	'Ingles_impreso', 'LIJ_digital', 'LIJ_impreso', 'Religion_digital','Religion_impreso', 'Texto_complementario_digital', 'Texto_complementario_impreso', 'Texto_curricular_digital',	'Texto_curricular_impreso',
'year','month','day',
       'day_week','week_number']]

###############################################################

datos['monto_Prediction'] = reg.predict(X_all)

datos

#########Plot de comparación##################################################

datos[['monto','monto_Prediction']].plot(figsize=(15, 5))


#####################################################################


####################métrica
mean_squared_error(y_true=datos['monto'],
                   y_pred=datos['monto_Prediction'])

##############################

mean_absolute_error(y_true=datos['monto'],
                   y_pred=datos['monto_Prediction'])

##############Porcentual##################
def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#############################################
mean_absolute_percentage_error(y_true=datos['monto'],
                   y_pred=datos['monto_Prediction'])


############Error de ajuste#########################


datos['residuo'] = abs(datos['monto_Prediction']-datos['monto'])


datos

error_residuo = sum(datos['residuo'])/sum(datos['monto'])

error_residuo

###################################################

# pickling the model 
import pickle 
pickle_out = open("C:/Users/hpalacios/Downloads/archive/forecast.pkl", "wb") 
pickle.dump(reg.predict, pickle_out) 
pickle_out.close()


####################################

loaded_model_x = pickle.load(open("C:/Users/hpalacios/Downloads/archive/forecast.pkl", "rb"))
# make predictions for test data
y_pred = loaded_model_x(X_all)

y_pred
