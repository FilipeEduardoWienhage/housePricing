import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score, accuracy_score
import plotly.express as px


# Gerar dados
def gerarDadosCasa(n_samples = 100):
    np.random.seed(42)
    size = np.random.normal(1500, 500, n_samples)
    price = size * 100 + np.random.normal(0, 10000, n_samples)
    return pd.DataFrame({'size_sqft': size, 'price': price})


# Treinamento

def modelo_treinamento():
    df = gerarDadosCasa()

    x = df[['size_sqft']]
    y = df['price']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(x_train, y_train)

    return model

def avalindo_modelo():
    r2_score = 0
    print(r2_score)
    accuracy_score = 0
    print(accuracy_score)
    root_mean_squared_error = 0
    print(root_mean_squared_error)

st.title('Predições de valores de Casas')
st.write('Digite o tamanho da casa em pés, para ser calculado o valor')

# ponto chave: utlizar o modelo

modeloUsado = modelo_treinamento()

size = st.number_input('Tamanho da casa em pés')

if st.button('Predição'):
    
    prediction = modeloUsado.predict([[size]])

    st.success(f'Preço estimado: $ {prediction[0]:.2f}')

    # Visualização

    df = gerarDadosCasa()
    fig = px.scatter(df,
                     x= 'size_sqft',
                     y= 'price',
                     title= 'Tamanho X Preço'
                     ) 
    fig.add_scatter(x=[size], y=[prediction[0]], mode= 'markers', marker=dict(size=15, color= 'red'), name = 'Predição')

    st.plotly_chart(fig)

