FROM python:3.8

WORKDIR /usr/src/BrandDetector

COPY . .

RUN pip install .