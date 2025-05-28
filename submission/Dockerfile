# Use official Miniconda base
FROM continuumio/miniconda3

# Set working directory
WORKDIR /workspace

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Update and install only necessary tools in one layer, then clean
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gnupg \
        gcc \
        build-essential \
        libc6-dev && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32 && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 871920D1991BC93C && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    apt-get update && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Configure conda to use TUNA mirrors
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --set show_channel_urls yes

# Copy and create Conda environment
COPY environment.yml .
RUN conda env create -f environment.yml && \
    conda clean --all --yes && \
    rm -rf /opt/conda/pkgs

# Copy project files and entrypoint
COPY . /workspace/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set entrypoint and default command
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
