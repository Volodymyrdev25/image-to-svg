FROM python:3.11-slim

ARG APP_HOME=/data

RUN adduser --disabled-password --gecos '' app_user

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR ${APP_HOME}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app_user:app_user . ${APP_HOME}

RUN chown -R app_user:app_user ${APP_HOME} && \
    mkdir -p /data/.EasyOCR && chown -R app_user:app_user /data/.EasyOCR

USER app_user

ENV EASYOCR_HOME=/home/app_user/.EasyOCR

RUN mkdir -p $EASYOCR_HOME && chown -R app_user:app_user $EASYOCR_HOME

RUN python -c "import easyocr; easyocr.Reader(['en'], download_enabled=True)"

ENTRYPOINT ["python", "/data/src/pipeline.py"]
