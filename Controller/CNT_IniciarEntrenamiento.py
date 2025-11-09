import os
import pandas as pd
from metadata import DATOS_ENTRENAMIENTO,ARCHIVO_SALIDA
from Controller.CNT_cargarJson import CargaJSON




    
def CargarData():
    # Lista de archivos
    archivos = ["BaresCotelesBucaramanga.json", "HotelesBucaramanga.json", "otrosBaresBogota.json","OtrosBaresBucaramanga.json","PH_Bucaramanga.json"]

    # Diccionario de etiquetas personalizadas
    etiquetas = {
        "BaresCotelesBucaramanga.json": "ZonaBares",
        "HotelesBucaramanga.json": "ZonaResidencial",
        "otrosBaresBogota.json": "ZonaOtrosBares",
        "OtrosBaresBucaramanga.json": "ZonaOtrosBares",
        "PH_Bucaramanga.json": "ZonaResidencial",
    }

    # Crear objeto
    cargador = CargaJSON(archivos, carpeta=DATOS_ENTRENAMIENTO)

    # Cargar todo
    datos = cargador.cargar()

    # 1. Extraemos todas las claves únicas de resumen de TODOS los archivos
    resumen_keys = set()
    for nombre_archivo in archivos:
        contenido = datos[nombre_archivo]
        for registro in contenido:
            resumen_keys.update(registro.get("resumen", {}).keys())
    
    # 2. Definir columnas estandarizadas
    columnas = ["nombre", "lat", "lon"] + sorted(resumen_keys) + ["tipo"]
    
    # 3. Construimos filas manualmente procesando TODOS los archivos
    filas = []
    for nombre_archivo in archivos:
        contenido = datos[nombre_archivo]
        
        for registro in contenido:
            fila = [
                registro.get("nombre"),
                registro.get("lat"),
                registro.get("lon")
            ]
            # Añadir los valores de cada clave de resumen en el mismo orden
            resumen = registro.get("resumen", {})
            for key in sorted(resumen_keys):
                fila.append(resumen.get(key, 0))  # 0 si no existe
            
            # Etiqueta inicial según archivo
            etiqueta = etiquetas.get(nombre_archivo, "Desconocido")

            # 👉 Validación especial: si tipo == "K", cambia etiqueta
            if registro.get("tipo") == "K":
                etiqueta = "ZonaBares"

            fila.append(etiqueta)
            filas.append(fila)
    
    # 4. Crear DataFrame (opcional para debug/CSV)
    df = pd.DataFrame(filas, columns=columnas)

    nombre_csv = "datosEntrenamiento.csv"
    ruta_csv = os.path.join(ARCHIVO_SALIDA, nombre_csv)
    df.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    
    print(f"✅ Procesados {len(archivos)} archivos con {len(filas)} registros totales")
    print(f"📄 CSV guardado en: {ruta_csv}")