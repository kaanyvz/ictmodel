# Base image with PyTorch, CUDA, and Python 3.9
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Install Python 3.9 and other dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    curl && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    curl https://bootstrap.pypa.io/get-pip.py | python3.9

# Set working directory
WORKDIR /app

# Copy model and app files
COPY ./model /app/model
COPY app.py /app
COPY requirements.txt /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 8080

# Start the Flask application
CMD ["python3", "app.py"]