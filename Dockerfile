FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

ENTRYPOINT ["python", "-m", "google.cloud.aiplatform.prediction.model_server"]

# The directory is created by root. This sets permissions so that any user can
# access the folder.
RUN mkdir -m 777 -p /usr/app /home
WORKDIR /usr/app
ENV HOME=/home

RUN apt update
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN apt install python3-pip -y
RUN apt install ffmpeg -y
RUN pip install --no-cache-dir --force-reinstall 'google-cloud-aiplatform[prediction]>=1.27.0'

ENV HANDLER_MODULE=google.cloud.aiplatform.prediction.handler

ENV HANDLER_CLASS=PredictionHandler

ENV PREDICTOR_MODULE=faster_whisper_predictor

ENV PREDICTOR_CLASS=FasterWhisperPredictor


# Copy requirements first
COPY ./requirements.txt ./
RUN pip install --no-cache-dir --force-reinstall -r requirements.txt

# copy the rest of the files
COPY [".", "."]


