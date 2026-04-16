# 使用官方 Python 3.11 基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 先复制 requirements.txt 并安装依赖
# 这样可以更好地利用 Docker 的缓存层
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有项目文件到工作目录
COPY . .

# 启动命令：因为我们在根目录，所以需要进入 backend 文件夹启动
CMD cd backend && python app.py