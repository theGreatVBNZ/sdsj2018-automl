FROM python:3.6

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN apt update -y && apt install vim -y

WORKDIR /cnt

RUN jupyter lab
