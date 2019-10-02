
FROM ubuntu:18.04

ENV ND_ENTRYPOINT="/startup/startup.sh"
RUN export ND_ENTRYPOINT="/startup/startup.sh" \
    && apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           apt-utils \
           bzip2 \
           ca-certificates \
           curl \
           locales \
           unzip \
	   emacs \
	   python2.7 \
	   python3.6 \
	   python3.7 \
	   wget \
	   libopenblas-base \
	   gcc \
	   unzip \
	   git \
	   cmake-curses-gui \
	   libglu1 \
	   qt5-default \
	   zlib1g \
	   libqt5opengl5 \
	   libglu1-mesa free\
	   glut3 \
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

ENV ANTSPATH="/opt/ants-2.3.1" \
    PATH="/opt/ants-2.3.1:$PATH"
RUN echo "Downloading ANTs ..." \
    && mkdir -p /opt/ants-2.3.1 \
    && curl -fsSL --retry 5 https://dl.dropbox.com/s/1xfhydsf4t4qoxg/ants-Linux-centos6_x86_64-v2.3.1.tar.gz \
    | tar -xz -C /opt/ants-2.3.1 --strip-components 1

ENV C3DPATH="/opt/convert3d-1.0.0" \
    PATH="/opt/convert3d-1.0.0/bin:$PATH"
RUN echo "Downloading Convert3D ..." \
    && mkdir -p /opt/convert3d-1.0.0 \
    && curl -fsSL --retry 5 https://sourceforge.net/projects/c3d/files/c3d/1.0.0/c3d-1.0.0-Linux-x86_64.tar.gz/download \
    | tar -xz -C /opt/convert3d-1.0.0 --strip-components 1

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

ENV ITKSNAPPATH="/opt/itksnap" \
    PATH="/opt/itksnap:$PATH"
RUN echo "Downloading ITK-SNAP ..." \
    && mkdir -p /opt/itksnap \
    && wget -P /opt/itksnap/ http://www.nitrc.org/frs/downloadlink.php/11442

ENV FREESURFERPATH="/opt/freesurfer" \
    PATH="/opt/freesurfer/bin:$PATH"
RUN echo "Downloading FreeSurfer ..." \
    && mkdir -p /opt/freesurfer \
    && wget -P /opt/freesurfer/ https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/dev/freesurfer-linux-centos6_x86_64-dev.tar.gz \
    && tar -xvzf /opt/freesurfer/freesurfer-linux-centos6_x86_64-dev.tar.gz -C /opt/freesurfer --strip-components 1

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




