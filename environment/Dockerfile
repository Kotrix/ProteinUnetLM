# hash:sha256:d4ff5ffd5f5c8a7684960d9f4697ba3b1bb041069744547d417d44bb0030d60c
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        perl=5.30.0-9ubuntu0.2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir --upgrade-strategy=only-if-needed \
    numpy==1.21.0 \
    pandas==1.2.3 \
    pycm==3.6 \
    sentencepiece==0.1.97 \
    tensorflow-gpu==2.9.1

COPY postInstall /
RUN /postInstall
