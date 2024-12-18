FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

LABEL email="su8939@skku.edu"
LABEL name="Seongwook Kim (COMPASSLAB SKKU)"
LABEL version="1.0"
LABEL description="This dockerfile is for building image for TDS Simulator"

RUN apt-get update && \
    apt-get install -y init sudo && \
    apt-get clean all

WORKDIR "/root"

COPY . /root/tds-sim

CMD ["sleep", "infinity"]
