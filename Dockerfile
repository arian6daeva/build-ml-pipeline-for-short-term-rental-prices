# docker build -t udacity_env/latest -f Dockerfile .
# docker run -i -t udacity_env/latest /bin/bash

FROM continuumio/miniconda3:latest

COPY . /home/udacity/

WORKDIR /home/udacity/

RUN conda env create -f environment.yml

RUN conda activate nyc_airbnb_dev