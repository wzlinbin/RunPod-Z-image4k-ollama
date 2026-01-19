# 阶段一：从官方镜像提取推理引擎和预下载模型
FROM ollama/ollama:latest AS ollama-base
RUN ollama serve & sleep 5 && ollama pull huihui_ai/qwen3-vl-abliterated

# 阶段二：构建主运行镜像 (参考信息 1)
FROM runpod/worker-comfyui:5.5.1-base

# 1. 拷贝 Ollama 二进制和完整的 GPU 推理库 (复刻昨日成功路径)
COPY --from=ollama-base /usr/bin/ollama /usr/bin/ollama
COPY --from=ollama-base /usr/lib/ollama /usr/lib/ollama
COPY --from=ollama-base /root/.ollama /root/.ollama

# 2. 安装 ComfyUI 必要依赖 (参考信息 1)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir transformers==4.47.0 accelerate==0.34.0 requests runpod

# 3. 下载 z-image-turbo 相关模型 (参考信息 1)
RUN comfy model download --url https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors --relative-path models/vae --filename ae.safetensors && \
    comfy model download --url https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors --relative-path models/diffusion_models --filename z_image_turbo_bf16.safetensors && \
    comfy model download --url https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors --relative-path models/clip --filename qwen_3_4b.safetensors

# 4. 设置 A40 专用环境变量 (完全复刻昨日成功配置)
ENV OLLAMA_LIBRARY_PATH=/usr/lib/ollama \
    LD_LIBRARY_PATH=/usr/lib/ollama:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=127.0.0.1:11434 \
    TMPDIR=/comfyui/tmp

# 5. 修复驱动链接 (完全复刻昨日成功指令)
RUN ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so && \
    ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so

# 6. 设置目录与权限
RUN mkdir -p /comfyui/tmp /comfyui/output && chmod 1777 /comfyui/tmp /comfyui/output

WORKDIR /comfyui
COPY pre_start.sh runpod_handler.py ./
RUN chmod +x pre_start.sh

CMD ["/bin/bash", "/comfyui/pre_start.sh"]
