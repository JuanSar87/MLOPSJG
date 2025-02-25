# Imagen base con Python y bibliotecas de ML
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar archivos del proyecto
COPY . /app

# Instalar dependencias
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Comando por defecto
CMD uvicorn main:app --host 0.0.0.0 --port 8080 --reload & jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser

