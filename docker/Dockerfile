# Use an official NVIDIA CUDA runtime image as a parent image
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install system dependencies including python3-pip
# Using -qq for quieter output and combining apt-get calls to reduce layers
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the application code (infer.py) into the container
# 'infer.py' is at the root of the build context (which is set to ./docker in the GH Action)
COPY infer.py /app/infer.py

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    faster-whisper \
    Flask \
    Flask-CORS \
    gunicorn

# Make port 5000 available
EXPOSE 5000

# Command to run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "infer:app"]
