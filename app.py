
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

import streamlit as st

st.title("Predicción de Producción con Prophet")

# Subir archivo Excel
uploaded_file = st.file_uploader("Subí el archivo Excel (.xlsm)", type=["xlsm"])
if uploaded_file is not None:
    try:
        # Leer hoja específica con encabezado en la segunda fila
        sheet_name = "Bajada Produccion - ORACLE"
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=1, engine="openpyxl")

        # Verificar columnas necesarias
        required_columns = ['Descripcion del Producto', 'Fecha de Produccion', 'Litros']
        if not all(col in df.columns for col in required_columns):
            st.error("Faltan columnas necesarias en el archivo Excel.")
        else:
            # Convertir fechas y eliminar nulos
            df['Fecha de Produccion'] = pd.to_datetime(df['Fecha de Produccion'], errors='coerce')
            df = df.dropna(subset=['Fecha de Produccion', 'Litros'])

            # Crear carpeta de resultados
            os.makedirs("resultados", exist_ok=True)

            resultados = {}

            # Iterar por producto
            for producto in df['Descripcion del Producto'].unique():
                df_prod = df[df['Descripcion del Producto'] == producto][['Fecha de Produccion', 'Litros']].copy()
                df_prod.columns = ['ds', 'y']

                promedio = df_prod['y'].mean()
                if pd.isna(promedio) or promedio == 0:
                    continue

                limite_inferior = 0.8 * promedio
                limite_superior = 1.2 * promedio
                df_filtrado = df_prod[(df_prod['y'] >= limite_inferior) & (df_prod['y'] <= limite_superior)]

                if len(df_filtrado) >= 10:
                    try:
                        modelo = Prophet()
                        modelo.fit(df_filtrado)

                        futuro = modelo.make_future_dataframe(periods=365)
                        forecast = modelo.predict(futuro)

                        forecast_12m = forecast[forecast['ds'] > df_filtrado['ds'].max()]
                        forecast_12m = forecast_12m.set_index('ds').resample('M').sum().reset_index()
                        forecast_12m['valor_estimado'] = forecast_12m['yhat'] * 10

                        nombre_archivo = f"resultados/prediccion_12m_{producto[:30].replace('/', '_')}.csv"
                        forecast_12m.to_csv(nombre_archivo, index=False)

                        resultados[producto] = forecast_12m
                    except Exception as e:
                        st.warning(f"No se pudo procesar el producto '{producto}': {e}")

            # Volumen global mensual
            volumen_global = df.set_index('Fecha de Produccion')['Litros'].resample('M').sum().reset_index()
            volumen_global.to_csv("resultados/volumen_global_planta.csv", index=False)

            # Graficar volumen global
            plt.figure(figsize=(10, 5))
            plt.plot(volumen_global['Fecha de Produccion'], volumen_global['Litros'], marker='o')
            plt.title("Volumen Global de Producción por Mes")
            plt.xlabel("Mes")
            plt.ylabel("Litros")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("resultados/volumen_global_predicho.png")

            st.success("Predicciones mensuales y volumen global generados correctamente.")
            st.image("resultados/volumen_global_predicho.png")

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")
else:
    st.info("Esperando que subas un archivo Excel para comenzar.")


