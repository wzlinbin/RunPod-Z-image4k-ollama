import runpod
import requests
import time
import os
import base64

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
COMFY_URL = "http://127.0.0.1:8188"

def handler(job):
    job_input = job.get("input", {})
    
    if "image" in job_input:
        # ... (保持原有的视觉识别逻辑不变)
        pass

    else:
        prompt_text = job_input.get("prompt", "a beautiful girl, 4k resolution, high detail")
        output_dir = "/comfyui/output"
        old_files = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()

        # 修改后的工作流：加入放大节点
        workflow = {
            "39": {"inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "default"}, "class_type": "CLIPLoader"},
            "40": {"inputs": {"vae_name": "ae.safetensors"}, "class_type": "VAELoader"},
            # 1. 基础生成分辨率设为 1024 (保证构图)
            "41": {"inputs": {"width": 1024, "height": 1024, "batch_size": 1}, "class_type": "EmptySD3LatentImage"},
            "45": {"inputs": {"text": prompt_text, "clip": ["39", 0]}, "class_type": "CLIPTextEncode"},
            "42": {"inputs": {"conditioning": ["45", 0]}, "class_type": "ConditioningZeroOut"},
            "46": {"inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
            "47": {"inputs": {"shift": 3.0, "model": ["46", 0]}, "class_type": "ModelSamplingAuraFlow"},
            "44": {"inputs": {"seed": int(time.time()), "steps": 9, "cfg": 1.0, "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0, "model": ["47", 0], "positive": ["45", 0], "negative": ["42", 0], "latent_image": ["41", 0]}, "class_type": "KSampler"},
            
            # 2. 关键步骤：潜空间放大到 4096
            "50": {
                "inputs": {
                    "upscale_method": "nearest-exact", 
                    "width": 4096, 
                    "height": 4096, 
                    "crop": "disabled", 
                    "samples": ["44", 0]
                }, 
                "class_type": "LatentUpscaleBy"
            },
            
            # 3. 对放大后的图像进行微调（Denoise 设低，如 0.3-0.4）防止糊掉
            "51": {
                "inputs": {
                    "seed": int(time.time()), "steps": 5, "cfg": 1.0, "sampler_name": "res_multistep", "scheduler": "simple", 
                    "denoise": 0.35, "model": ["47", 0], "positive": ["45", 0], "negative": ["42", 0], "latent_image": ["50", 0]
                }, 
                "class_type": "KSampler"
            },

            "43": {"inputs": {"samples": ["51", 0], "vae": ["40", 0]}, "class_type": "VAEDecode"},
            "9": {"inputs": {"filename_prefix": "z-image-4k", "images": ["43", 0]}, "class_type": "SaveImage"}
        }

        try:
            res = requests.post(f"{COMFY_URL}/prompt", json={"prompt": workflow})
            res.raise_for_status()
            prompt_id = res.json().get("prompt_id")
            
            # 增加超时等待时间，因为 4K 渲染很慢
            for _ in range(300): 
                history_res = requests.get(f"{COMFY_URL}/history/{prompt_id}").json()
                if prompt_id in history_res:
                    break
                time.sleep(2)

            new_files = set(os.listdir(output_dir)) - old_files
            if new_files:
                target = sorted([f for f in new_files if f.startswith("z-image-4k")])[-1]
                with open(os.path.join(output_dir, target), "rb") as f:
                    return {"status": "success", "type": "generation", "image": base64.b64encode(f.read()).decode("utf-8")}
            return {"status": "error", "message": "No output files found."}
        except Exception as e:
            return {"status": "error", "message": f"4K 生成失败: {str(e)}"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
