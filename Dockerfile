FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Montreal

RUN apt-get update && apt-get install -y apt-utils apt-transport-https git wget zip build-essential cmake vim screen
RUN apt-get remove python-* && apt-get autoremove
RUN apt-get install -y python3 python3-dev python3-pip python-is-python3 

# Install latest exiftool
RUN wget https://exiftool.org/Image-ExifTool-12.70.tar.gz &&\
    gzip -dc Image-ExifTool-*.tar.gz | tar -xf - && \
    cd Image-ExifTool-* && \
    perl Makefile.PL && \
    make install

ADD . /mercury-ducking

RUN cd /mercury-ducking && pip install -r requirements.txt
WORKDIR /mercury-ducking

CMD ["tail", "-f", "/dev/null"]
