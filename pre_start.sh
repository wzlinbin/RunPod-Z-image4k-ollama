#!/bin/bash
echo "🚀 [Pre-start] 正在清理并初始化 GPU 环境..."

# 1. 强制释放显存幽灵进程
fuser -k /dev/nvidia0

# 2. 环境变量强制加载 (完全复刻昨日成功配置)
export OLLAMA_LIBRARY_PATH=/usr/lib/ollama
export LD_LIBRARY_PATH=/usr/lib/ollama:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

# 3. 关键：禁止 ComfyUI 吞噬所有显存
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4. 架构修复 (参考信息 2)
find /usr/local/lib/python3.10/dist-packages/transformers -name "*.pyc" -delete
find /comfyui -name "*.pyc" -delete
cd /comfyui && git fetch --all && git reset --hard origin/master

# 5. 启动 Ollama
ollama serve > /var/log/ollama.log 2>&1 &

# 6. 健康检查
python3 -c "import requests, time; 
for i in range(30):
    try:
        r = requests.get('http://127.0.0.1:11434/api/tags')
        if r.status_code == 200:
            print('✅ Ollama 已接管 GPU'); break
    except: pass
    time.sleep(2)
"

# 7. 启动 ComfyUI (加上低显存模式参数，防止抢占)
python /comfyui/main.py --listen 127.0.0.1 --port 8188 --lowvram > /var/log/comfyui.log 2>&1 &

# 8. 启动主 Handler
python -u /comfyui/runpod_handler.py
