# 使用 Debian slim 基础镜像（比 Alpine 兼容性好，体积适中）
FROM python:3.11-slim

WORKDIR /app

# 可选：安装 OpenCV 所需的最小系统依赖（很多情况下不需要）
# 如果不需要，可以删除这一整段
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装 Python 包（CPU 版 PyTorch）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 复制项目文件（排除 models/best_model.pth，因为启动时会自动下载）
COPY backend/ backend/
COPY frontend/ frontend/
COPY models/config.json models/

# 启动后端服务
CMD cd backend && python app.py