FROM python:3.10-slim

WORKDIR /app

# Встановлення системних залежностей
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Встановлення Python-бібліотек
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіювання коду експерименту та даних
COPY train.py .
COPY data/ ./data/

CMD ["python", "train.py"]