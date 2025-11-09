import requests
import os
import json
from collections import Counter
from metadata import ARCHIVO_CARGA, ARCHIVO_SALIDA
import sys
import pandas as pd



# ------------------------------
# Configuración
# ------------------------------
INPUT_FILE = os.path.join(ARCHIVO_CARGA, "Establecimiento de Gastronomía y Bar Bogota.json")  # archivo con coordenadas
OUTPUT_FILE = os.path.join(ARCHIVO_SALIDA, "resultadodePubtos.json")  # salida
print('INPUT_FILE ' +INPUT_FILE)
# Etiquetas principales funcionales
main_keys = [
    "amenity", "shop", "building", "highway", "leisure",
    "healthcare", "railway", "tourism", "man_made", "landuse"
]

# Radio de búsqueda
radio = 500

# ------------------------------
# Función que consulta Overpass
# ------------------------------
def consultar_osm(lat, lon):
    query = f"""
    [out:json][timeout:25];
    (
      way["highway"](around:{radio},{lat},{lon});
      way["building"](around:{radio},{lat},{lon});
      node["shop"](around:{radio},{lat},{lon});
      node["amenity"="restaurant"](around:{radio},{lat},{lon});
      node["amenity"="cafe"](around:{radio},{lat},{lon});
      node["amenity"="fast_food"](around:{radio},{lat},{lon});
      node["amenity"="bar"](around:{radio},{lat},{lon});
      node["amenity"="hospital"](around:{radio},{lat},{lon});
      node["amenity"="clinic"](around:{radio},{lat},{lon});
      node["amenity"="pharmacy"](around:{radio},{lat},{lon});
      node["amenity"="school"](around:{radio},{lat},{lon});
      node["amenity"="college"](around:{radio},{lat},{lon});
      node["amenity"="university"](around:{radio},{lat},{lon});
      way["leisure"="park"](around:{radio},{lat},{lon});
      way["leisure"="pitch"](around:{radio},{lat},{lon});
      node["amenity"="bus_station"](around:{radio},{lat},{lon});
      node["highway"="bus_stop"](around:{radio},{lat},{lon});
      node["railway"="station"](around:{radio},{lat},{lon});
      node["amenity"="place_of_worship"](around:{radio},{lat},{lon});
    );
    out body;
    >;
    out skel qt;
    """
    url = "https://overpass-api.de/api/interpreter"
    response = requests.post(url, data={"data": query})
    return response.json()

# ------------------------------
# Procesar un solo punto
# ------------------------------
def procesar_punto(punto):
    props = punto.get("properties", {})
    nombre = props.get("NomEstable", "Sin nombre")
    lat = props.get("Latitud")
    lon = props.get("Longitud")
    tipo = props.get("SubCatRNT")
    # lat = punto.get("latitud")
    # lon = punto.get("longitud")



    print(f"📍 Consultando punto: {nombre} ({lat}, {lon})")
    if not lat or not lon:
        print(f"⚠️ Coordenadas inválidas para {nombre}")
        return None

    try:
        data = consultar_osm(lat, lon)
    except Exception as e:
        print(f"❌ Error al consultar {nombre}: {e}")
        return None

    contador = Counter()
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        main_tag = next((k for k in main_keys if k in tags), None)
        if main_tag:
            val_clean = tags[main_tag].replace("_", " ").title()
            contador[val_clean] += 1

    return {
        "nombre": nombre,
        "typo":tipo,
        "lat": lat,
        "lon": lon,
        "resumen": dict(contador)
    }

# ------------------------------
# Función principal
# ------------------------------
def nearbySearch():
    extension = os.path.splitext(INPUT_FILE)[1].lower()

    # Lista que almacenará los puntos
    puntos = []

    if extension == ".json":
        # Leer archivo JSON
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            puntos = json.load(f)

    elif extension == ".csv":
        # Leer archivo CSV
        df = pd.read_csv(INPUT_FILE, encoding="utf-8")

        # Convertir cada fila en un diccionario con latitud y longitud
        for _, row in df.iterrows():
            
            punto = {
                "latitud": row["latitud"],
                "longitud": row["longitud"],
                "name": row["Direccion"]
            }
            
            puntos.append(punto)

    else:
        raise ValueError(f"❌ Formato de archivo no soportado: {extension}")

    # Procesar cada punto
    resultados = []
  
    for punto in puntos:
        resultado = procesar_punto(punto)
        if resultado:
            resultados.append(resultado)

    # Guardar el resultado
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print("✅ Proceso completado con éxito.")

