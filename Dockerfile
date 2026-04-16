# 1. 使用更小的 Alpine 基础镜像
FROM python:3.11-alpine

# 2. 安装运行 OpenCV 等所需的系统依赖
RUN apk add --no-cache --virtual .build-deps gcc musl-dev linux-headers \
    && apk add --no-cache libstdc++ libgcc libx11 libxext libxrender \
    libxcb libxau libxdmcp libjpeg-turbo libpng libwebp libtiff libx264 \
    && apk add --no-cache openblas-dev

WORKDIR /app

# 3. 复制依赖文件并安装 Python 包 (CPU-only)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 4. 复制项目文件（排除 models/ 等大文件）
COPY backend/ backend/
COPY frontend/ frontend/
COPY models/config.json models/

# 5. 创建 entrypoint 脚本，用于在容器启动时下载模型
RUN echo '#!/bin/sh' > /entrypoint.sh \
    && echo 'python -c "from model import get_classifier; get_classifier()"' >> /entrypoint.sh \
    && chmod +x /entrypoint.sh

# 6. 运行后端服务
CMD cd backend && python app.py