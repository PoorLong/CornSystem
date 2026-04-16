# 使用 Debian slim 基础镜像
FROM python:3.11-slim

WORKDIR /app

# 复制依赖文件并安装 Python 包（CPU 版 PyTorch）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 复制项目文件（模型文件不打包，运行时自动下载）
COPY backend/ backend/
COPY frontend/ frontend/
COPY models/config.json models/

# 启动后端服务
CMD ["python", "backend/app.py"]