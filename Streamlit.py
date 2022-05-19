import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

df = pd.read_csv('housing.csv')

if st.button('Отобразить первые пять строк'):
    st.write(df.head())
    
test_size = st.selectbox(
    'Необходимо выбрать размер test-выборки',
    ('-', 10, 15, 20, 25, 30, 35))
st.write(f'Размер test-выборки составит: {test_size} %')

if st.button('Обучить модель'):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('MEDV', axis=1),
                                                        df['MEDV'],
                                                        test_size=test_size/100,
                                                        random_state=2100)
    st.write('Разделили данные и передали в обучение')
    regr_model = XGBRegressor()
    regr_model.fit(X_train, y_train)
    pred = regr_model.predict(X_test)    
    st.write('Обучили модель, MAE = ' + str(mean_absolute_error(y_test, pred)))
    
    list_pred = list(zip(y_test, pred))
    df_pred = pd.DataFrame(list_pred, columns=['y_test', 'pred'])
    st.dataframe(df_pred)
    
