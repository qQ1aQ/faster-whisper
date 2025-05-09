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

# Copy the local requirements file and application code into the container
# We'll copy infer.py and any other necessary files like jfk.flac (if used by default)
COPY ./docker/infer.py /app/infer.py
# If jfk.flac is an example file for infer.py and needs to be in the image:
# COPY ./docker/jfk.flac /app/jfk.flac 
# If jfk.flac is NOT needed by the server at runtime, you can remove the above line.
# For now, assuming infer.py might reference it as it did in the original Dockerfile context

# Install Python dependencies
# Consider using a requirements.txt file for more complex projects
RUN pip3 install --no-cache-dir \
    faster-whisper \
    Flask \
    Flask-CORS \
    gunicorn

# Make port 5000 available to the world outside this container (if using gunicorn default port)
# Runpod will handle the actual mapping from its public port to this container port.
EXPOSE 5000

# Define environment variables if needed, e.g., for model configuration
# ENV MODEL_NAME="tiny"

# Command to run the application using Gunicorn
# Binds to 0.0.0.0 to be accessible from outside the container (within Runpod's network)
# 'infer:app' means Gunicorn should look for an object named 'app' in the 'infer.py' module.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "infer:app"]
