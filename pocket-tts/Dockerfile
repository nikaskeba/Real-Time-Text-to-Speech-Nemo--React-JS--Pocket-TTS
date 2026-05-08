FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY gateway-requirements.txt /app/gateway-requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /app/gateway-requirements.txt

COPY gateway.py /app/gateway.py

EXPOSE 9080

CMD ["python", "/app/gateway.py"]
