# OliviaPrompt Web（模型推理走 llama.cpp 服务容器，见 docker-compose.yml）
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

EXPOSE 7860
CMD ["olivia-prompt", "web", "--bind", "0.0.0.0:7860"]
