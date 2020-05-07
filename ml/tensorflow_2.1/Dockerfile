
FROM ubuntu:18.04

ENV ND_ENTRYPOINT="/startup/startup.sh"
RUN export ND_ENTRYPOINT="/startup/startup.sh" \
    && apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           apt-utils \
           locales \
	   emacs \
	   python2.7 \
	   python3.5 \
	   python3-pip \
	   wget \
	   gcc \
	   unzip \
	   git \
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

RUN pip3 install tensorflow==2.1
RUN pip3 install pandas
RUN pip3 install matplotlib



