FROM ubuntu:18.04

ENV LANG="en_US.UTF-8" \
    LC_ALL="en_US.UTF-8" \
    ND_ENTRYPOINT="/docker/startup.sh"
RUN export ND_ENTRYPOINT="/docker/startup.sh" \
    && apt-get update -qq \
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
    && chmod 777 /opt && chmod a+s /opt \
    && mkdir -p /docker \
    && if [ ! -f "$ND_ENTRYPOINT" ]; then \
         echo '#!/usr/bin/env bash' >> "$ND_ENTRYPOINT" \
    &&   echo 'set -e' >> "$ND_ENTRYPOINT" \
    &&   echo 'export USER="${USER:=`whoami`}"' >> "$ND_ENTRYPOINT" \
    &&   echo 'if [ -n "$1" ]; then "$@"; else /usr/bin/env bash; fi' >> "$ND_ENTRYPOINT"; \
    fi \
    && chmod -R 777 /docker && chmod a+s /docker

ENTRYPOINT ["/docker/startup.sh"]

ENV ANTSPATH="/opt/ants-2.3.1" \
    PATH="/opt/ants-2.3.1:$PATH"
RUN echo "Downloading ANTs ..." \
    && mkdir -p /opt/ants-2.3.1 \
    && curl -fsSL --retry 5 https://dl.dropbox.com/s/1xfhydsf4t4qoxg/ants-Linux-centos6_x86_64-v2.3.1.tar.gz \
    | tar -xz -C /opt/ants-2.3.1 --strip-components 1

ENV PATH="/opt/dcm2niix-latest/bin:$PATH"
RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           cmake \
           g++ \
           gcc \
           git \
           make \
           pigz \
           zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && git clone https://github.com/rordenlab/dcm2niix /tmp/dcm2niix \
    && mkdir /tmp/dcm2niix/build \
    && cd /tmp/dcm2niix/build \
    && cmake  -DCMAKE_INSTALL_PREFIX:PATH=/opt/dcm2niix-latest .. \
    && make \
    && make install \
    && rm -rf /tmp/dcm2niix


ENV C3DPATH="/opt/convert3d-1.0.0" \
    PATH="/opt/convert3d-1.0.0/bin:$PATH"
RUN echo "Downloading Convert3D ..." \
    && mkdir -p /opt/convert3d-1.0.0 \
    && curl -fsSL --retry 5 https://sourceforge.net/projects/c3d/files/c3d/1.0.0/c3d-1.0.0-Linux-x86_64.tar.gz/download \
    | tar -xz -C /opt/convert3d-1.0.0 --strip-components 1

ENV ITKSNAPPATH="/opt/itksnap" \
    PATH="/opt/itksnap:$PATH"
RUN echo "Downloading ITK-SNAP ..." \
    && mkdir -p /opt/itksnap \
    && wget -P /opt/itksnap/ https://sourceforge.net/projects/itk-snap/files/itk-snap/Nightly/itksnap-nightly-master-Linux-gcc64.tar.gz \
    && tar -xvzf /opt/itksnap/itksnap-nightly-master-Linux-gcc64.tar.gz -C /opt/itksnap --strip-components 1

ENV FSLDIR="/opt/fsl-6.0.1" \
    PATH="/opt/fsl-6.0.1/bin:$PATH"
RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           bc \
           dc \
           file \
           libfontconfig1 \
           libfreetype6 \
           libgl1-mesa-dev \
           libglu1-mesa-dev \
           libgomp1 \
           libice6 \
           libxcursor1 \
           libxft2 \
           libxinerama1 \
           libxrandr2 \
           libxrender1 \
           libxt6 \
           wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && echo "Downloading FSL ..." \
    && mkdir -p /opt/fsl-6.0.1 \
    && wget -P /opt/fsl-6.0.1/ https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-6.0.1-centos6_64.tar.gz \
    && tar -xvzf /opt/fsl-6.0.1/fsl-6.0.1-centos6_64.tar.gz -C /opt/fsl-6.0.1 --strip-components 1 \
    && sed -i '$isource $FSLDIR/etc/fslconf/fsl.sh' $ND_ENTRYPOINT


RUN pip3 install numpy==1.18.4
RUN pip3 install pandas==1.0.3
RUN pip3 install scipy==1.4.1
RUN pip3 install nibabel
RUN pip3 install dipy
RUN pip3 install fury
RUN pip3 install setuptools==46.1.3
RUN pip3 install nipy
RUN pip3 install nilearn
RUN pip3 install seaborn
RUN pip3 install matplotlib
RUN pip3 install pandas
RUN pip3 install scikit-learn
RUN pip3 install pingouin

RUN pip3 install requests==2.23.0
RUN pip3 install deprecation==2.1.0
RUN pip3 install mtspec==0.3.2
RUN pip3 install xlrd

RUN pip install pennprov==2.2.4
RUN pip install setuptools
RUN pip3 install bctpy
RUN pip3 install spyder_kernels
RUN pip3 install jupyter
RUN pip3 install nipype
WORKDIR /home/

RUN git clone https://github.com/andyrevell/echobase.git

WORKDIR /home/echobase/ieegpy
RUN python2.7 setup.py build
RUN python3.6 setup.py install
WORKDIR /mnt
RUN mkdir -p /root/.local/share/jupyter/runtime

