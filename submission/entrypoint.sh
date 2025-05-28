#!/bin/bash
# 初始化 conda 环境
source /opt/conda/etc/profile.d/conda.sh
conda activate segvol
exec "$@"