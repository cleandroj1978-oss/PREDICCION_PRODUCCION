
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
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from io import BytesIO

# Configuración de la página
st.set_page_config(page_title="Predicción de Producción", layout="wide")

st.title("📊 Predicción de Producción por Producto y Volumen Global")

# Cargar archivo Excel
uploaded_file = st.file_uploader("Subí el archivo Excel de producción", type=["xls", "xlsx", "xlsm"])
if uploaded_file:
    try:
        sheet_name = "Bajada Produccion - ORACLE"
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=1, engine="openpyxl")

        # Verificar columnas necesarias
        required_columns = ['Descripcion del Producto', 'Fecha de Produccion', 'Litros']
        if not all(col in df.columns for col in required_columns):
            st.error("❌ El archivo no contiene las columnas necesarias.")
        else:
            # Preprocesamiento
            df['Fecha de Produccion'] = pd.to_datetime(df['Fecha de Produccion'], errors='coerce')
            df = df.dropna(subset=['Fecha de Produccion', 'Litros'])

            # Filtro por producto
            productos = df['Descripcion del Producto'].unique()
            producto_seleccionado = st.selectbox("Seleccioná un producto para ver su predicción", productos)

            # Filtrar datos del producto
            df_prod = df[df['Descripcion del Producto'] == producto_seleccionado][['Fecha de Produccion', 'Litros']].copy()
            df_prod.columns = ['ds', 'y']

            # Aplicar filtro 80%-120% del promedio
            promedio = df_prod['y'].mean()
            if promedio > 0:
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

                    st.subheader(f"📈 Predicción para: {producto_seleccionado}")
                    fig1, ax1 = plt.subplots(figsize=(10, 4))
                    ax1.plot(forecast_12m['ds'], forecast_12m['yhat'], marker='o')
                    ax1.set_title(f"Predicción mensual de litros para {producto_seleccionado}")
                    ax1.set_xlabel("Mes")
                    ax1.set_ylabel("Litros")
                    ax1.grid(True)
                    st.pyplot(fig1)

                    st.dataframe(forecast_12m)

                    # Botón para descargar CSV
                    csv_buffer = BytesIO()
                    forecast_12m.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Descargar predicción del producto en CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"prediccion_12m_{producto_seleccionado[:30]}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("⚠️ No hay suficientes datos filtrados para este producto.")
            else:
                st.warning("⚠️ El promedio de litros es cero o inválido.")

            # Predicción volumen global
            st.subheader("🌍 Predicción del Volumen Global de Producción")
            volumen_global = df.set_index('Fecha de Produccion')['Litros'].resample('M').sum().reset_index()

            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(volumen_global['Fecha de Produccion'], volumen_global['Litros'], marker='o', color='green')
            ax2.set_title("Volumen Global de Producción por Mes")
            ax2.set_xlabel("Mes")
            ax2.set_ylabel("Litros")
            ax2.grid(True)
            st.pyplot(fig2)

            # Descargar volumen global
            csv_global = BytesIO()
            volumen_global.to_csv(csv_global, index=False)
            st.download_button(
                label="📥 Descargar volumen global en CSV",
                data=csv_global.getvalue(),
                file_name="volumen_global_planta.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")
else:
    st.info("📁 Esperando que subas un archivo Excel para comenzar.")


