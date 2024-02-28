FROM nvcr.io/nvidia/pytorch:20.12-py3

WORKDIR /workspace
ENV PYTHONPATH "${PYTHONPATH}:/opt/conda/bin/python"

# Setup tools for the nethack learning environment
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    autoconf \
    libtool \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-numpy \
    git \
    flex \
    bison \
    libbz2-dev \
    software-properties-common \
    libgl1 && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
    apt-get --allow-unauthenticated install -y cmake kitware-archive-keyring

# Install requirements
COPY requirements.txt .
# ToDo Is there a cleaner way to handle the PyYAML error? https://stackoverflow.com/questions/49911550/how-to-upgrade-disutils-package-pyyaml
RUN pip install -r requirements.txt --ignore-installed PyYAML

# Install our customized nle-language-wrapper
# RUN pip install -e nle-language-wrapper[dev]

# Install our project in developer mode
COPY . .
RUN pip install -e .