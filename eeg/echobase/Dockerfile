
FROM ubuntu:18.04

ENV ND_ENTRYPOINT="/startup/startup.sh"
RUN export ND_ENTRYPOINT="/startup/startup.sh" \
    && apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           apt-utils \
           locales \
	   emacs \
	   #python2.7 \
	   #python3.7 \
	   python3-pip \
	   python-pip \
	   wget \
	   gcc \
	   unzip \
	   git \
	   gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /startup \
    && mkdir -p /mount \
    && mkdir -p /scripts \
    && if [ ! -f "$ND_ENTRYPOINT" ]; then \
         echo '#!/usr/bin/env bash' >> "$ND_ENTRYPOINT" \
    &&   echo 'set -e' >> "$ND_ENTRYPOINT" \
    &&   echo 'export USER="${USER:=`whoami`}"' >> "$ND_ENTRYPOINT" \
    &&   echo 'if [ -n "$1" ]; then "$@"; else /usr/bin/env bash; fi' >> "$ND_ENTRYPOINT"; \
    fi \
    && chmod -R 777 /startup && chmod a+s /startup

ENTRYPOINT ["/startup/startup.sh"]

RUN pip3 install numpy==1.18.4
RUN pip3 install pandas==1.0.3
RUN pip3 install scipy==1.4.1
RUN pip3 install requests==2.23.0
RUN pip3 install deprecation==2.1.0
RUN pip3 install setuptools==46.1.3
RUN pip3 install mtspec==0.3.2

RUN pip install pennprov==2.2.4
RUN pip install setuptools
#apt-get install gfortran is for mtspec python package
#pennprov: for ieegpy downloading ieeg.org data
#requests: for ieegpy downloading ieeg.org data
#deprecation: for ieegpy downloading ieeg.org data
#to install pennprov, need python 3.6 or lower. So that is why installing python 2.7 with pip rather than pip3.

#copying relevant pipeline python code to docker image. Need to have these files (and ONLY THESE FILES) in the same directory as Dockerfile (going to copy everything in this directory...)
#COPY * /home/

#WORKDIR /home/ieegpy
#RUN python2.7 /home/ieegpy/setup.py build
#RUN python3.6 /home/ieegpy/setup.py install

WORKDIR /home/

RUN git clone https://github.com/andyrevell/paper001.git

WORKDIR /home/paper001/pipelines/ieegpy
RUN python2.7 setup.py build
RUN python3.6 setup.py install
WORKDIR /home/paper001/pipelines/