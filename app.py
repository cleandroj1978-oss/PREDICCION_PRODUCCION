import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from io import BytesIO

# Título de la app
st.title("Predicción de Producción - Sherwin Williams")

# Cargar archivo Excel
uploaded_file = st.file_uploader("Subí el archivo Excel exportado desde Power BI", type=["xls", "xlsx", "xlsm"])

if uploaded_file:
    try:
        # Leer hoja específica
        df = pd.read_excel(uploaded_file, sheet_name='Bajada Produccion - ORACLE', engine='openpyxl')

        # Mostrar preview de datos
        st.subheader("Vista previa de los datos")
        st.dataframe(df.head())

        # Verificar columnas necesarias
        if 'PRODUCTO' in df.columns and 'FECHA' in df.columns and 'CANTIDAD' in df.columns:
            # Selección de producto
            productos = df['PRODUCTO'].unique()
            producto_seleccionado = st.selectbox("Seleccioná un producto para predecir", productos)

            # Filtrar datos
            df_producto = df[df['PRODUCTO'] == producto_seleccionado][['FECHA', 'CANTIDAD']]
            df_producto = df_producto.groupby('FECHA').sum().reset_index()
            df_producto.columns = ['ds', 'y']  # Prophet requiere estas columnas

            # Mostrar gráfico histórico
            st.subheader("Producción histórica")
            fig_hist, ax = plt.subplots()
            ax.plot(df_producto['ds'], df_producto['y'], marker='o')
            ax.set_title(f"Producción histórica de {producto_seleccionado}")
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Cantidad")
            st.pyplot(fig_hist)

            # Entrenar modelo Prophet
            modelo = Prophet()
            modelo.fit(df_producto)

            # Crear fechas futuras
            futuro = modelo.make_future_dataframe(periods=365)
            forecast = modelo.predict(futuro)

            # Mostrar gráfico de predicción
            st.subheader("Predicción de producción")
            fig_pred = modelo.plot(forecast)
            st.pyplot(fig_pred)

            # Descargar resultados
            st.subheader("Descargar predicción en CSV")
            forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            csv = forecast_result.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar CSV",
                data=csv,
                file_name=f'prediccion_{producto_seleccionado}.csv',
                mime='text/csv'
            )
        else:
            st.error("Las columnas necesarias ('PRODUCTO', 'FECHA', 'CANTIDAD') no están presentes en la hoja.")
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
else:
    st.info("Esperando que subas el archivo Excel...")
