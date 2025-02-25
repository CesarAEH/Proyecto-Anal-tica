#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
df4 = pd.read_excel("datosfinalesapartamentos.xlsx")  


# In[20]:


#Se crea el archivo con todas las variables transformadas a numericas
# Convertir 'state' en variables dummies numéricas
state_dummies = pd.get_dummies(df4["state"], prefix="state").astype(int)

# Convertir 'has_photo' a valores binarios 
df4["has_photo"] = df4["has_photo"].apply(lambda x: 1 if x == "Yes" else 0)
# Convertir 'source' 
source_dummies = pd.get_dummies(df4["source"], prefix="source").astype(int)

# Unir las nuevas variables al DataFrame original
df4 = pd.concat([df4, source_dummies], axis=1)

# Eliminar la columna original de 'source'
df4 = df4.drop(columns=["source"])

top_cities = df4["cityname"].value_counts().head(10).index

# Crear una nueva columna donde agrupamos las ciudades menos comunes como 'Other'
df4["cityname_grouped"] = df4["cityname"].apply(lambda x: x if x in top_cities else "Other")


# Convertir 'cityname' en variables dummies numéricas
cityname_dummies = pd.get_dummies(df4["cityname"], prefix="cityname").astype(int)
# Unir las dummies al DataFrame original
df4 = pd.concat([df4, state_dummies, cityname_dummies], axis=1)

df4["pets_allowed_Cats"] = df4["pets_allowed"].apply(lambda x: 1 if "Cats" in x else 0)
df4["pets_allowed_Dogs"] = df4["pets_allowed"].apply(lambda x: 1 if "Dogs" in x else 0)
# Eliminar las columnas originales de 'state' y 'cityname'
df4 = df4.drop(columns=["state", "cityname"])


# In[40]:


import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import statsmodels.api as sm



# Definir variables predictoras
variables_numericas = ["square_feet", "bathrooms", "bedrooms", "has_photo"]
categorias = list(cityname_dummies.columns) + list(state_dummies.columns)  # Ciudades y estados

X = df4[variables_numericas + ["pets_allowed_Cats", "pets_allowed_Dogs"] + categorias + list(source_dummies.columns)]
y = df4["price"]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



# modelo Lasso
model_lasso = Lasso(alpha=0.02)
model_lasso.fit(X_train, y_train)





# Obtener coeficientes 
coeficientes_importantes = pd.Series(model_lasso.coef_, index=X_train.columns).sort_values(ascending=False)
coef_ciudades = coeficientes_importantes.filter(like="cityname")
coef_estados = coeficientes_importantes.filter(like="state")

# Definir función para predecir precios
def predict_price(n_clicks, square_feet, bathrooms, bedrooms, has_photo, pets, state, city):
    if n_clicks is None:
        return "Esperando predicción..."

    new_apartment_dict = {col: 0 for col in X_train.columns}
    new_apartment_dict.update({
        "square_feet": square_feet,
        "bathrooms": bathrooms,
        "bedrooms": bedrooms,
        "has_photo": 1 if has_photo else 0,
        "pets_allowed_Cats": 1 if "pets_allowed_Cats" in pets else 0,
        "pets_allowed_Dogs": 1 if "pets_allowed_Dogs" in pets else 0,
        state: 1,
        city: 1
    })

    new_apartment_df = pd.DataFrame([new_apartment_dict]).reindex(columns=X_train.columns, fill_value=0)
    predicted_price = model_lasso.predict(new_apartment_df)[0]

    return f"Precio estimado: ${predicted_price:.2f} al mes"

# Diseñar el tablero Dash
app = dash.Dash(__name__, external_stylesheets=["https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"])

app.layout = html.Div([
    html.H1("Calculadora de Precios de Renta", style={'textAlign': 'center'}),

    # Entrada de datos
    html.Div([
        html.Label("Metros cuadrados:"),
        dcc.Input(id="square_feet", type="number", value=800, min=10, step=10),

        html.Label("Baños:"),
        dcc.Dropdown(id="bathrooms", options=[{'label': i, 'value': i} for i in range(1, 5)], value=2),

        html.Label("Habitaciones:"),
        dcc.Dropdown(id="bedrooms", options=[{'label': i, 'value': i} for i in range(1, 5)], value=2),

        html.Label("Tiene foto en el anuncio:"),
        dcc.Checklist(id="has_photo", options=[{'label': 'Sí', 'value': 1}], value=[1]),

        html.Label("Mascotas permitidas:"),
        dcc.Checklist(id="pets", options=[
            {'label': 'Gatos', 'value': "pets_allowed_Cats"},
            {'label': 'Perros', 'value': "pets_allowed_Dogs"}
        ], value=[]),

        html.Label("Estado:"),
        dcc.Dropdown(id="state", options=[{'label': col, 'value': col} for col in state_dummies.columns], value="state_CA"),

        html.Label("Ciudad:"),
        dcc.Dropdown(id="city", options=[{'label': col, 'value': col} for col in cityname_dummies.columns], value="cityname_Los Angeles"),

        html.Button("Calcular Precio", id="calcular_btn", n_clicks=0, className="btn btn-primary"),
    ], style={"width": "50%", "margin": "auto"}),

    # Salida del Precio Estimado
    html.Div(id="resultado", style={"textAlign": "center", "fontSize": 24, "margin": "20px"}),

    # Sección de Gráficos
    html.Div([
        dcc.Graph(id="grafico_importancia"),
        dcc.Graph(id="grafico_ciudades"),
        dcc.Graph(id="grafico_estados"),
        dcc.Graph(id="grafico_mascotas"),
        dcc.Graph(id="grafico_foto")
    ])
])

# Callbacks de Dash
@app.callback(
    [Output("resultado", "children"),
     Output("grafico_importancia", "figure"),
     Output("grafico_ciudades", "figure"),
     Output("grafico_estados", "figure"),
     Output("grafico_mascotas", "figure"),
     Output("grafico_foto", "figure")],
    [Input("calcular_btn", "n_clicks")],
    [dash.dependencies.State("square_feet", "value"),
     dash.dependencies.State("bathrooms", "value"),
     dash.dependencies.State("bedrooms", "value"),
     dash.dependencies.State("has_photo", "value"),
     dash.dependencies.State("pets", "value"),
     dash.dependencies.State("state", "value"),
     dash.dependencies.State("city", "value")]
)
def actualizar_tablero(n_clicks, square_feet, bathrooms, bedrooms, has_photo, pets, state, city):
    if n_clicks == 0:
        return "Esperando predicción...", {}, {}, {}, {}, {}

    resultado_precio = predict_price(n_clicks, square_feet, bathrooms, bedrooms, has_photo, pets, state, city)

    # Gráficos
    fig_importancia = px.bar(
        coeficientes_importantes.head(10), 
        title="Factores más influyentes en el precio",
        labels={"index": "Factor", "value": "Impacto en el precio (USD)"},
        color_discrete_sequence=["#636EFA"])
    
    fig_ciudades = px.bar(
        coef_ciudades.head(5), 
        title="Ciudades más costosas para rentar",
        labels={"index": "Ciudad", "value": "Impacto en el precio (USD)"},
        color_discrete_sequence=["#EF553B"]
    )


    fig_estados = px.bar(
        coef_estados.head(5), 
        title="Estados más costosos para rentar",
        labels={"index": "Estado", "value": "Impacto en el precio (USD)"},
        color_discrete_sequence=["#00CC96"]
    )


    fig_mascotas = px.bar(
        x=["Gatos", "Perros"], 
        y=[coeficientes_importantes["pets_allowed_Cats"], coeficientes_importantes["pets_allowed_Dogs"]],
        title="Impacto de permitir mascotas en el precio",
        labels={"x": "Tipo de mascota", "y": "Aumento en el precio (USD)"},
        color_discrete_sequence=["#AB63FA"]
    )


    fig_foto = px.bar(
        x=["Con foto", "Sin foto"], 
        y=[coeficientes_importantes["has_photo"], 0], 
        title="Impacto de la foto en el anuncio",
        labels={"x": "Foto en el anuncio", "y": "Aumento en el precio (USD)"},
        color_discrete_sequence=["#FFA15A"]
    )

    return resultado_precio, fig_importancia, fig_ciudades, fig_estados, fig_mascotas, fig_foto

# Ejecutar la app
if __name__ == '__main__':
    app.run_server(debug=True)

