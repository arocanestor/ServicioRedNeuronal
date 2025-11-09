import json
import os

class CargaJSON:
    def __init__(self, lista_archivos, carpeta=""):
        self.lista_archivos = lista_archivos
        self.carpeta = carpeta
    
    def cargar(self):
        resultados = {}
        for archivo in self.lista_archivos:
            ruta = os.path.join(self.carpeta, archivo)
            try:
                with open(ruta, "r", encoding="utf-8") as f:
                    resultados[archivo] = json.load(f)
            except Exception as e:
                resultados[archivo] = {"error": str(e)}
        return resultados