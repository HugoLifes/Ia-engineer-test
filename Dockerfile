# Usar una imagen base de Python con la versión adecuada
FROM python:3.9-slim-buster

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos de requisitos al contenedor
COPY requirements.txt .

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de los archivos de tu proyecto al contenedor
COPY . .

# Especificar el comando que se ejecutará al iniciar el contenedor
CMD ["python", "recomendation-spoti.py"]