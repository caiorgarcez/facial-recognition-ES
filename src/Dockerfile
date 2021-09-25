FROM python:3.8

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /code

COPY requirements.txt .

RUN apt-get update && apt-get -y install cmake protobuf-compiler
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt

COPY . .