from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from typing import Optional


import sys
import os
from performance_middleware import performance_logger

app = FastAPI()


from main import invocarRedNeuronal

# Agrega el middleware
app.middleware("http")(performance_logger)

# Crear la aplicación FastAPI
app = FastAPI(title="API de Direcciones")

# Modelo de entrada (más flexible)
class DireccionRequest(BaseModel):
    direccion: str = Field(default="", description="Dirección a procesar")

# Modelo de respuesta
class DireccionResponse(BaseModel):
    Error: str = Field(..., description="Mensaje de error o vacío si no hay error")
    Success: bool = Field(..., description="Indica si la operación fue exitosa")
    Estadisticas: str = Field(..., description="Información o estadísticas de la dirección")

# Endpoint POST principal
@app.post("/procesar-direccion/", response_model=DireccionResponse)
async def procesar_direccion(request: DireccionRequest):
    """
    Procesa una dirección y retorna estadísticas.
    
    Ejemplo de body:
    {
        "direccion": "Calle 123 #45-67"
    }
    """
    try:
        # Validar que la dirección no esté vacía
        if not request.direccion or request.direccion.strip() == "":
            return DireccionResponse(
                Error="La dirección no puede estar vacía",
                Success=False,
                Estadisticas=""
            )
        print( f"Dirección : {request.direccion}")
        # Aquí va tu lógica de procesamiento
        direccion = request.direccion.strip()
        
        # Ejemplo de procesamiento: generar estadísticas
        longitud = len(direccion)
        palabras = len(direccion.split())
        tiene_numero = any(char.isdigit() for char in direccion)
        
        estadisticas = f"Longitud: {longitud} caracteres, Palabras: {palabras}, Contiene números: {'Sí' if tiene_numero else 'No'}"
        Prediccion = invocarRedNeuronal(request.direccion)
        
        # Respuesta exitosa
        return DireccionResponse(
            Error="",
            Success=True,
            Estadisticas=Prediccion
        )
    
    except Exception as e:
        # Respuesta en caso de error
        return DireccionResponse(
            Error=f"Error al procesar la dirección: {str(e)}",
            Success=False,
            Estadisticas=""
        )

# Endpoint POST en la raíz (para evitar el error 405)
@app.post("/", response_model=DireccionResponse)
async def procesar_direccion_root(request: DireccionRequest):
    """
    Mismo endpoint pero en la raíz /
    Redirige al procesamiento principal
    """
    return await procesar_direccion(request)

# Endpoint GET raíz para verificar que la API está funcionando
@app.get("/")
async def root():
    return {
        "mensaje": "API de procesamiento de direcciones",
        "estado": "funcionando",
        "endpoints": {
            "POST /": "Procesa una dirección (raíz)",
            "POST /procesar-direccion/": "Procesa una dirección",
            "POST /validar-direccion/": "Valida una dirección",
            "GET /docs": "Documentación interactiva",
            "GET /test": "Endpoint de prueba"
        }
    }

# Endpoint de prueba simple
@app.get("/test")
async def test():
    return DireccionResponse(
        Error="",
        Success=True,
        Estadisticas="API funcionando correctamente"
    )

# Endpoint para debug - ver qué estás enviando
@app.post("/debug/")
async def debug_request(request: Request):
    """
    Endpoint para ver exactamente qué datos estás enviando
    """
    body = await request.body()
    return {
        "body_raw": body.decode(),
        "content_type": request.headers.get("content-type"),
        "method": request.method,
        "url": str(request.url)
    }

# Endpoint adicional con manejo de errores más específico
@app.post("/validar-direccion/", response_model=DireccionResponse)
async def validar_direccion(request: DireccionRequest):
    """
    Valida si una dirección cumple con ciertos criterios.
    """
    try:
        direccion = request.direccion.strip()
        
        if not direccion:
            return DireccionResponse(
                Error="La dirección está vacía",
                Success=False,
                Estadisticas=""
            )
        
        # Criterios de validación
        criterios_cumplidos = []
        criterios_no_cumplidos = []
        
        if len(direccion) >= 10:
            criterios_cumplidos.append("Longitud adecuada")
        else:
            criterios_no_cumplidos.append("Dirección muy corta")
        
        if any(char.isdigit() for char in direccion):
            criterios_cumplidos.append("Contiene número de calle")
        else:
            criterios_no_cumplidos.append("Falta número de calle")
        
        if len(direccion.split()) >= 2:
            criterios_cumplidos.append("Múltiples componentes")
        else:
            criterios_no_cumplidos.append("Dirección incompleta")
        
        # Generar estadísticas
        estadisticas = f"Criterios cumplidos: {len(criterios_cumplidos)}/3. "
        if criterios_cumplidos:
            estadisticas += f"✓ {', '.join(criterios_cumplidos)}. "
        if criterios_no_cumplidos:
            estadisticas += f"✗ {', '.join(criterios_no_cumplidos)}"
        
        # Determinar si es exitoso
        es_exitoso = len(criterios_no_cumplidos) == 0
        
        return DireccionResponse(
            Error="" if es_exitoso else "La dirección no cumple todos los criterios",
            Success=es_exitoso,
            Estadisticas=estadisticas
        )
    
    except Exception as e:
        return DireccionResponse(
            Error=f"Error en validación: {str(e)}",
            Success=False,
            Estadisticas=""
        )