import pandas as pd
import requests
import time
import os
from metadata import ARCHIVO_CARGA, ARCHIVO_SALIDA, API_KEY_GOOGLE

# Tu API Key de Google Maps
API_KEY = API_KEY_GOOGLE
INPUT_FILE = os.path.join(ARCHIVO_CARGA, "prop_horizontal_bogota_marzo2023.csv")  # archivo con coordenadas
OUTPUT_FILE = os.path.join(ARCHIVO_SALIDA, "puntosGoogle.csv")  # salida

# Función para llamar a la API de Geocoding y obtener latitud y longitud
def geocodeAddress(address):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": API_KEY,
        "region": "CO"  # para especificar Colombia y mejores resultados
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    print(data)
    if data['status'] == 'OK':
        location = data['results'][0]['geometry']['location']
        return location['lat'], location['lng']
    else:
        # Devuelve None si no encuentra resultado
        return None, None

def convertirDireccion():
    # Leer archivo CSV con columna "direccion"
    df = pd.read_csv(INPUT_FILE, delimiter=';')

    # Crear columnas para latitud y longitud
    df['latitud'] = None
    df['longitud'] = None

    # Iterar filas y geocodificar cada dirección
    for idx, row in df.iterrows():
        direccion = row['Direccion']
        lat, lng = geocodeAddress(direccion)
        
        df.at[idx, 'latitud'] = lat
        df.at[idx, 'longitud'] = lng
        
        # Para no exceder límite de peticiones (Google recomienda 50 peticiones/s máximo, pero mejor ir despacio)
        time.sleep(0.1)

    # Guardar resultados en nuevo CSV
    df.to_csv(OUTPUT_FILE, index=False)

    print("Proceso finalizado. Archivo generado: direcciones_con_coordenadas.csv")
