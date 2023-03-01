FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04

USER root

RUN apt-get update
RUN apt-get install -y python3-pip python3-dev
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN apt-get install -y apt-utils\
                        curl \
                        ca-certificates \
                        sudo \
                        git \
                        bzip2 \
                        libx11-6

 # Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user

# Installing all the default JAX stuff
RUN pip3 install numpy scipy pandas cmake matplotlib tqdm seaborn wandb
RUN pip3 install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip3 install chex equinox mctx optax

# Creating Projects folder in home folder of user
ENV HOME=/home/user
RUN chmod 777 /home/user
ENV PATH=$PATH:/home/user/.local/bin

USER user
WORKDIR /home/user

