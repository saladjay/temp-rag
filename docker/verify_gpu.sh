#!/usr/bin/env bash
# 探测 NVIDIA Container Toolkit 是否就绪，并校验 Milvus 健康
set -e
echo "== 宿主机 GPU =="
nvidia-smi || { echo "未检测到 nvidia-smi，请确认已安装 NVIDIA 驱动"; exit 1; }

echo "== Docker GPU 运行时 =="
if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
  echo "未检测到 nvidia runtime，请安装 NVIDIA Container Toolkit："
  echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
  exit 1
fi
docker run --rm --gpus all milvusdb/milvus:v2.4.10-gpu nvidia-smi >/dev/null 2>&1 \
  && echo "容器内 GPU 可用" || echo "警告：容器内 GPU 不可用"

echo "== Milvus 健康 =="
curl -sf http://localhost:9092/healthz >/dev/null && echo "Milvus 健康" \
  || echo "Milvus 未就绪（先执行：docker compose -f docker/docker-compose-gpu.yml up -d）"
