import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Título de la app
st.title("Predicción de Producción - Sherwin Williams")

# Cargar archivo Excel
uploaded_file = st.file_uploader("Subí el archivo Excel exportado desde Power BI", type=["xls", "xlsx", "xlsm"])

if uploaded_file:
    try:
        # Leer hoja específica con encabezado en la segunda fila
        df = pd.read_excel(uploaded_file, sheet_name='Bajada Produccion - ORACLE', header=1, engine='openpyxl')

        # Mostrar preview de datos
        st.subheader("Vista previa de los datos")
        st.dataframe(df.head())

        # Verificar columnas necesarias
        if 'Producto' in df.columns and 'Fecha de Produccion' in df.columns and 'Litros' in df.columns:
            # Selección múltiple de productos
            productos = df['Producto'].unique()
            productos_seleccionados = st.multiselect("Seleccioná uno o más productos para predecir", productos)

            for producto in productos_seleccionados:
                # Filtrar datos
                df_producto = df[df['Producto'] == producto][['Fecha de Produccion', 'Litros']]
                df_producto = df_producto.groupby('Fecha de Produccion').sum().reset_index()
                df_producto.columns = ['ds', 'y']  # Prophet requiere estas columnas

                # Mostrar gráfico histórico
                st.subheader(f"Producción histórica - {producto}")
                fig_hist, ax = plt.subplots()
                ax.plot(df_producto['ds'], df_producto['y'], marker='o')
                ax.set_title(f"Producción histórica de {producto}")
                ax.set_xlabel("Fecha")
                ax.set_ylabel("Litros")
                st.pyplot(fig_hist)

                # Entrenar modelo Prophet
                modelo = Prophet()
                modelo.fit(df_producto)

                # Crear fechas futuras
                futuro = modelo.make_future_dataframe(periods=365)
                forecast = modelo.predict(futuro)

                # Mostrar gráfico de predicción
                st.subheader(f"Predicción de producción - {producto}")
                fig_pred = modelo.plot(forecast)
                st.pyplot(fig_pred)

                # Descargar resultados
                st.subheader(f"Descargar predicción en CSV - {producto}")
                forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                csv = forecast_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Descargar CSV - {producto}",
                    data=csv,
                    file_name=f'prediccion_{producto}.csv',
                    mime='text/csv'
                )
        else:
            st.error("Las columnas necesarias ('Producto', 'Fecha de Produccion', 'Litros') no están presentes en la hoja.")
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
else:
    st.info("Esperando que subas el archivo Excel...")


