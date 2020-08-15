FROM ubuntu:16.04

ARG DEBIAN_FRONTEND="noninteractive"

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
	   cmake-curses-gui \
	   libglu1 \
	   qt5-default \
	   zlib1g \
	   libqt5opengl5 \
	   libglu1-mesa \ 
	   freeglut3 \
	   libmuparser-dev \
           libqalculate-dev \
           libqt5svg5-dev \
           libqt5x11extras5-dev \
           qtbase5-dev \
           qtdeclarative5-dev \
    	   libfontconfig1 \
	   libboost-dev \
	   mesa-common-dev \
    	   libgl1-mesa-dev \
    	   libglu1-mesa-dev \
    	   freeglut3-dev \
    	   mesa-utils \
	   zlib1g \
  	   libqt5opengl5 \
  	   libglu1-mesa \
	   xvfb \
	   build-essential \
           autoconf \
           libtool \
           pkg-config \
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
           libboost-all-dev \
           zlib1g-dev \
           freeglut3-dev \
           make \
           zlib1g-dev \
           software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure --frontend=noninteractive locales \
    && update-locale LANG="en_US.UTF-8" \
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


#DSI STUDIO
RUN add-apt-repository ppa:beineri/opt-qt-5.12.2-xenial \
    && apt-get update \
    && apt install -y --no-install-recommends \
    freetds-common libclang1-5.0 libllvm5.0 libodbc1 libsdl2-2.0-0 libsndio6.1 \
    libsybdb5 libxcb-xinerama0 qt5123d qt512base qt512canvas3d \
    qt512connectivity qt512declarative qt512graphicaleffects \
    qt512imageformats qt512location qt512multimedia qt512scxml qt512svg \
    qt512wayland qt512x11extras qt512xmlpatterns qt512charts-no-lgpl \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV QT_BASE_DIR="/opt/qt512"
ENV QTDIR="$QT_BASE_DIR" \
    PATH="$QT_BASE_DIR/bin:$PATH:/opt/dsi-studio/dsi_studio_64" \
    LD_LIBRARY_PATH="$QT_BASE_DIR/lib/x86_64-linux-gnu:$QT_BASE_DIR/lib:$LD_LIBRARY_PATH" \
    PKG_CONFIG_PATH="$QT_BASE_DIR/lib/pkgconfig:$PKG_CONFIG_PATH"
RUN mkdir /opt/dsi-studio \
  && cd /opt/dsi-studio \
  && git clone https://github.com/frankyeh/DSI-Studio.git \
  && mv DSI-Studio src \
  && git clone https://github.com/frankyeh/TIPL.git \
  && mv TIPL src/tipl \
  && mkdir build && cd build \
  && /opt/qt512/bin/qmake ../src && make \
  && cd /opt/dsi-studio \
  && curl -sSLO 'https://pitt.box.com/shared/static/n3ln8k7bakr7y5p0ekm3g4s0ucdmqkia.zip' \
  && unzip n3ln8k7bakr7y5p0ekm3g4s0ucdmqkia.zip \
  && rm n3ln8k7bakr7y5p0ekm3g4s0ucdmqkia.zip \
  && cd dsi_studio_64 \
  && rm *.dll \
  && rm *.exe \
  && rm -rf iconengines \
  && rm -rf imageformats \
  && rm -rf platforms \
  && rm -rf styles \
  && mv ../build/dsi_studio . \
  && rm -rf /opt/dsi-studio/src /opt/dsi-studio/build


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

ENV DSISTUDIOPATH="/opt/dsistudio" \
    PATH="/opt/dsistudio:$PATH"
RUN echo "Downloading DSI studio ..." \
    && mkdir -p /opt/dsistudio \
    && wget -P /opt/dsistudio/ http://www.lin4neuro.net/lin4neuro/neuroimaging_software_packages/dsistudio1804.zip \
    && unzip /opt/dsistudio/dsistudio1804.zip -d /opt/



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
WORKDIR /home/

RUN git clone https://github.com/andyrevell/echobase.git

WORKDIR /home/echobase/ieegpy
RUN python2.7 setup.py build
RUN python3.6 setup.py install
WORKDIR /mnt
RUN mkdir -p /root/.local/share/jupyter/runtime

