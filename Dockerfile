FROM nvcr.io/nvidia/pytorch:23.08-py3

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo  \
	cmake ninja-build && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

# Set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"

# Install additional requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# For debugging purposes
RUN pip install pdbpp ipython
COPY ./.pdbrc.py /.pdbrc.py

# Wandb: stores wandb authentication key, run `wandb login` if not present
COPY ./.netrc /root/.netrc

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/app/.torch/"

RUN alias python=python3

WORKDIR /app/code
