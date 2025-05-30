FROM python:3.11.12-slim

WORKDIR /app

# Install build dependencies for lightfm
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn
# Debug step to verify gunicorn installation
RUN which gunicorn && gunicorn --version

# Copy application code
COPY . .

# Run gunicorn with the correct port binding
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]
