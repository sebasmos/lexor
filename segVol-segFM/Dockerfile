# 使用官方Miniconda镜像作为基础
FROM continuumio/miniconda3

# 设置工作目录
WORKDIR /workspace

# 导入必要的GPG密钥
RUN apt-get update && apt-get install -y gnupg && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32 && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 871920D1991BC93C

# 创建并配置apt源
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-backports main restricted universe multiverse" >> /etc/apt/sources.list 

RUN apt-get update

# 安装依赖
RUN apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# 配置conda镜像
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --set show_channel_urls yes

# 添加环境文件并创建环境
COPY environment.yml .
# RUN conda env create -f environment.yml
RUN conda env create -f environment.yml --verbose

# 设置Shell以使用conda环境
SHELL ["conda", "run", "-n", "segvol", "/bin/bash", "-c"]

# 复制代码
COPY . /workspace/

# 设置默认启动命令
CMD ["sh", "predict.sh"]