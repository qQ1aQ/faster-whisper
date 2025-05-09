FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04
WORKDIR /root
RUN apt-get update -y && apt-get install -y python3-pip
COPY infer.py ./
RUN pip3 install faster-whisper Flask
EXPOSE 5000
CMD ["python3", "infer.py"]
