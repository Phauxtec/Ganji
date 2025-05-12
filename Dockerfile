FROM python:3.10-slim

# System setup
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Environment variables
ENV MODEL_NAME=csarron/mobilebert-uncased-squad-v2
ENV PORT=8000

# Expose API port
EXPOSE 8000

# Run the Flask app
CMD ["python", "app.py"]
