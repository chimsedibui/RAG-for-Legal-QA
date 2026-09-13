FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY core/ core/
COPY providers/ providers/
COPY services/ services/
COPY tools/ tools/
COPY api/ api/
COPY main.py ./
EXPOSE 8000
CMD ["python", "main.py"]
