FROM ubuntu:18.04

RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends --fix-missing \
           apt-utils \
           bzip2 \
           ca-certificates \
           curl \
           locales \
           python3-pip \
	   python-pip \
           unzip \
           wget \
	   libopenblas-base \
	   gcc \
           unzip \
           git \
           gfortran \
           emacs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
