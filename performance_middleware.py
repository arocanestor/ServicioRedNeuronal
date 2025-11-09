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
    response = await call_next(request)

    duration = time.time() - start_time
    cpu_percent = process.cpu_percent()
    memory_mb = process.memory_info().rss / (1024 * 1024)

    log_message = (
        f"{request.method} {request.url.path} | "
        f"Tiempo: {duration:.3f}s | "
        f"CPU: {cpu_percent:.1f}% | "
        f"RAM: {memory_mb:.2f} MB"
    )

    # Guarda en archivo y muestra en consola
    logger.info(log_message)
    print("[PERFORMANCE]", log_message)

    return response
