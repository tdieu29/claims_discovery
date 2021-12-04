# Base image
FROM python:3.7-slim

# Install dependencies
COPY setup.py setup.py
COPY requirements.txt requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e . --no-cache-dir \
    && apt-get purge -y --auto-remove gcc build-essential

# CONNECT WITH DATABSE SOMEHOW...
# Copy
COPY app app
COPY colbert colbert
COPY config config
COPY cord19_data cord19_data
COPY ETL ETL
COPY streamlit streamlit
COPY t5 t5

# Export port (8501  for streamlit app)
EXPOSE 8501

# Start app
ENTRYPOINT ["streamlit", "run", "streamlit/app.py"]
