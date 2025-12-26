# Use Python 3.9
FROM python:3.9-slim

# Install system dependencies for OpenCV and GLib
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create cache directory for DeepFace/YOLO to avoid permission errors
# Hugging Face runs as user 1000
RUN mkdir -p /app/.deepface && chmod -R 777 /app/.deepface
ENV DEEPFACE_HOME=/app/.deepface

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Command to run the app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app", "--timeout", "120"]
