
import importlib

required_packages = ['streamlit', 'prophet', 'matplotlib', 'openpyxl']
missing = []

for pkg in required_packages:
    if importlib.util.find_spec(pkg) is None:
        missing.append(pkg)

if missing:
    print(f"Faltan los siguientes paquetes: {', '.join(missing)}")
else:
    print("Todos los paquetes necesarios están instalados.")


import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

pip install scikit-learn

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.linear_model import LinearRegression
import numpy as np
import io

# Configuración de la página
st.set_page_config(page_title="Predicción de Producción", layout="wide")

st.title("📊 Predicción de Producción por Producto y Volumen Global")

# Cargar archivo Excel
uploaded_file = st.file_uploader("Subí el archivo Excel", type=["xls", "xlsx", "xlsm"])
if uploaded_file:
    sheet_name = "Bajada Produccion - ORACLE"
    try:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=1, engine="openpyxl")
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        st.stop()

    # Verificar columnas necesarias
    required_columns = ['Descripcion del Producto', 'Fecha de Produccion', 'Litros']
    if not all(col in df.columns for col in required_columns):
        st.error("Faltan columnas necesarias en el archivo Excel.")
        st.stop()

    # Preprocesamiento
    df['Fecha de Produccion'] = pd.to_datetime(df['Fecha de Produccion'], errors='coerce')
    df = df.dropna(subset=['Fecha de Produccion', 'Litros'])

    # Filtro por producto
    productos = df['Descripcion del Producto'].unique()
    producto_seleccionado = st.selectbox("Seleccioná un producto para ver su predicción", productos)

    df_prod = df[df['Descripcion del Producto'] == producto_seleccionado][['Fecha de Produccion', 'Litros']].copy()
    df_prod.columns = ['ds', 'y']

    # Filtrar valores dentro del rango 80%-120% del promedio
    promedio = df_prod['y'].mean()
    limite_inferior = 0.8 * promedio
    limite_superior = 1.2 * promedio
    df_filtrado = df_prod[(df_prod['y'] >= limite_inferior) & (df_prod['y'] <= limite_superior)]

    if len(df_filtrado) >= 10:
        modelo = Prophet()
        modelo.fit(df_filtrado)

        futuro = modelo.make_future_dataframe(periods=365)
        forecast = modelo.predict(futuro)

        forecast_12m = forecast[forecast['ds'] > df_filtrado['ds'].max()]
        forecast_12m = forecast_12m.set_index('ds').resample('M').sum().reset_index()
        forecast_12m['valor_estimado'] = forecast_12m['yhat'] * 10

        # Gráfico con etiquetas y línea de tendencia
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(forecast_12m['ds'], forecast_12m['yhat'], marker='o', label='Predicción')

        # Etiquetas de datos
        for i, row in forecast_12m.iterrows():
            ax.text(row['ds'], row['yhat'], f"{row['yhat']:.0f}", ha='center', va='bottom', fontsize=8)

        # Línea de tendencia
        x = np.arange(len(forecast_12m))
        y = forecast_12m['yhat'].values
        x_reshape = x.reshape(-1, 1)
        model = LinearRegression().fit(x_reshape, y)
        trend = model.predict(x_reshape)
        r2 = model.score(x_reshape, y)
        ax.plot(forecast_12m['ds'], trend, color='red', linestyle='--', label=f'Tendencia\ny={model.coef_[0]:.2f}x+{model.intercept_:.2f}\nR²={r2:.2f}')

        ax.set_title(f"Predicción mensual para {producto_seleccionado}")
        ax.set_xlabel("Mes")
        ax.set_ylabel("Litros")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # Descargar CSV
        csv = forecast_12m.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar predicción en CSV", data=csv, file_name=f"prediccion_{producto_seleccionado}.csv", mime='text/csv')
    else:
        st.warning("No hay suficientes datos filtrados para este producto.")

    # Volumen global
    volumen_global = df.set_index('Fecha de Produccion')['Litros'].resample('M').sum().reset_index()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(volumen_global['Fecha de Produccion'], volumen_global['Litros'], marker='o', label='Volumen Global')

    # Etiquetas de datos
    for i, row in volumen_global.iterrows():
        ax2.text(row['Fecha de Produccion'], row['Litros'], f"{row['Litros']:.0f}", ha='center', va='bottom', fontsize=8)

    # Línea de tendencia
    xg = np.arange(len(volumen_global))
    yg = volumen_global['Litros'].values
    xg_reshape = xg.reshape(-1, 1)
    modelg = LinearRegression().fit(xg_reshape, yg)
    trendg = modelg.predict(xg_reshape)
    r2g = modelg.score(xg_reshape, yg)
    ax2.plot(volumen_global['Fecha de Produccion'], trendg, color='green', linestyle='--', label=f'Tendencia\ny={modelg.coef_[0]:.2f}x+{modelg.intercept_:.2f}\nR²={r2g:.2f}')

    ax2.set_title("Volumen Global de Producción por Mes")
    ax2.set_xlabel("Mes")
    ax2.set_ylabel("Litros")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

    # Descargar CSV global
    csv_global = volumen_global.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar volumen global en CSV", data=csv_global, file_name="volumen_global.csv", mime='text/csv')
else:
    st.info("Por favor, subí un archivo Excel para comenzar.")


