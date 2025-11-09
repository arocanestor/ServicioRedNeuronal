import time
import psutil
import os
import logging
from fastapi import Request

# Configura logging (rotación diaria, nivel INFO)
from logging.handlers import TimedRotatingFileHandler

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "performance.log")

handler = TimedRotatingFileHandler(log_file, when="midnight", backupCount=7)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)

logger = logging.getLogger("performance")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Inicializa el proceso actual
process = psutil.Process(os.getpid())

async def performance_logger(request: Request, call_next):
    start_time = time.time()
    
    # Obtener métricas de I/O antes de procesar
    try:
        io_before = process.io_counters()
    except (psutil.AccessDenied, AttributeError):
        io_before = None
    
    response = await call_next(request)

    duration = time.time() - start_time
    
    # Métricas del sistema
    cpu_percent = process.cpu_percent()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)  # Memoria física (RSS)
    memory_virtual_mb = memory_info.vms / (1024 * 1024)  # Memoria virtual
    
    # Métricas de I/O
    if io_before:
        try:
            io_after = process.io_counters()
            io_read_mb = (io_after.read_bytes - io_before.read_bytes) / (1024 * 1024)
            io_write_mb = (io_after.write_bytes - io_before.write_bytes) / (1024 * 1024)
        except (psutil.AccessDenied, AttributeError):
            io_read_mb = "N/A"
            io_write_mb = "N/A"
    else:
        io_read_mb = "N/A"
        io_write_mb = "N/A"
    
    # Métricas de threads y conexiones
    num_threads = process.num_threads()
    try:
        num_connections = len(process.connections())
    except (psutil.AccessDenied, AttributeError):
        num_connections = "N/A"  # Puede requerir permisos especiales
    
    # Status code de la respuesta
    status_code = response.status_code if hasattr(response, 'status_code') else "N/A"
    
    # Tamaño del request body usando Content-Length header (más seguro)
    try:
        content_length = request.headers.get('content-length')
        request_size_kb = int(content_length) / 1024 if content_length else 0.0
    except (ValueError, TypeError):
        request_size_kb = 0.0

    # Formatear valores numéricos o N/A
    io_read_str = f"{io_read_mb:.2f} MB" if isinstance(io_read_mb, (int, float)) else f"{io_read_mb}"
    io_write_str = f"{io_write_mb:.2f} MB" if isinstance(io_write_mb, (int, float)) else f"{io_write_mb}"
    
    log_message = (
        f"{request.method} {request.url.path} | "
        f"Status: {status_code} | "
        f"Tiempo: {duration:.3f}s | "
        f"CPU: {cpu_percent:.1f}% | "
        f"RAM: {memory_mb:.2f} MB | "
        f"RAM_Virtual: {memory_virtual_mb:.2f} MB | "
        f"Threads: {num_threads} | "
        f"Conexiones: {num_connections} | "
        f"I/O_Read: {io_read_str} | "
        f"I/O_Write: {io_write_str} | "
        f"Request_Size: {request_size_kb:.2f} KB"
    )

    # Guarda en archivo y muestra en consola
    logger.info(log_message)
    print("[PERFORMANCE]", log_message)

    return response
