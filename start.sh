#!/bin/bash

# 设置环境变量
export PYTHONUNBUFFERED=1

# 创建必要的目录
mkdir -p /opt/render/project/src/cache

# 设置PyTorch缓存目录
export TORCH_HOME=/opt/render/project/src/cache

# 启动服务
uvicorn backend.api:app --host 0.0.0.0 --port $PORT
