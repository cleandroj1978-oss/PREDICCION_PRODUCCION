import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

# Cargar el archivo Excel y la hoja correspondiente
file_path = "PROYECTO DASHBOARD SW V1.XLSM"
sheet_name = "Bajada Produccion - ORACLE"

# Leer el archivo con encabezado en la segunda fila
df = pd.read_excel(file_path, sheet_name=sheet_name, header=1, engine="openpyxl")

# Verificar que las columnas necesarias existen
required_columns = ['Descripcion del Producto', 'Fecha de Produccion', 'Litros']
if not all(col in df.columns for col in required_columns):
    raise ValueError("Faltan columnas necesarias en el archivo Excel.")

# Convertir la fecha a formato datetime
df['Fecha de Produccion'] = pd.to_datetime(df['Fecha de Produccion'], errors='coerce')
df = df.dropna(subset=['Fecha de Produccion', 'Litros'])

# Crear un diccionario para almacenar resultados
resultados = {}

# Iterar por cada producto único
for producto in df['Descripcion del Producto'].dropna().unique():
    df_prod = df[df['Descripcion del Producto'] == producto][['Fecha de Produccion', 'Litros']].copy()
    df_prod.columns = ['ds', 'y']

    # Filtrar valores dentro del rango 80%-120% del promedio
    promedio = df_prod['y'].mean()
    if pd.isna(promedio) or promedio == 0:
        continue  # Saltar si el promedio no es válido

    limite_inferior = 0.8 * promedio
    limite_superior = 1.2 * promedio
    df_filtrado = df_prod[(df_prod['y'] >= limite_inferior) & (df_prod['y'] <= limite_superior)]

    # Continuar solo si hay suficientes datos
    if len(df_filtrado) >= 10:
        try:
            # Entrenar modelo Prophet
            modelo = Prophet()
            modelo.fit(df_filtrado)

            # Crear fechas futuras para 12 meses
            futuro = modelo.make_future_dataframe(periods=365)
            forecast = modelo.predict(futuro)

            # Filtrar predicción para los próximos 12 meses
            forecast_12m = forecast[forecast['ds'] > df_filtrado['ds'].max()]
            forecast_12m = forecast_12m.set_index('ds').resample('M').sum().reset_index()

            # Calcular valor estimado de producción (ejemplo: 10 unidades monetarias por litro)
            forecast_12m['valor_estimado'] = forecast_12m['yhat'] * 10

            # Guardar resultados
            resultados[producto] = forecast_12m
        except Exception as e:
            print(f"Error al procesar el producto {producto}: {e}")

# Calcular volumen global de la planta por mes
volumen_global = df.set_index('Fecha de Produccion')['Litros'].resample('M').sum().reset_index()

# Guardar resultados en archivos
os.makedirs("resultados_prediccion", exist_ok=True)
for producto, df_pred in resultados.items():
    nombre_archivo = f"resultados_prediccion/prediccion_12m_{producto[:30].replace('/', '_')}.csv"
    df_pred.to_csv(nombre_archivo, index=False)

volumen_global.to_csv("resultados_prediccion/volumen_global_planta.csv", index=False)

# Graficar volumen global
plt.figure(figsize=(10, 5))
plt.plot(volumen_global['Fecha de Produccion'], volumen_global['Litros'], marker='o')
plt.title("Volumen Global de Producción por Mes")
plt.xlabel("Mes")
plt.ylabel("Litros")
plt.grid(True)
plt.tight_layout()
plt.savefig("resultados_prediccion/volumen_global_predicho.png")

print("Predicciones mensuales y volumen global generados correctamente.")




