import os
import sys
import torch
from PIL import Image

# 设置 ComfyUI 模块路径（确保能找到 comfy、nodes、folder_paths 等模块）
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import comfy.utils
import comfy.model_management as model_management
import comfy.sd
import folder_paths
from comfy_extras.nodes_sd3 import (
    TripleCLIPLoader, 
    CLIPTextEncodeSD3, 
    EmptySD3LatentImage
)
from comfy_extras.nodes_model_advanced import ModelSamplingSD3
from nodes import (
    CLIPTextEncode,
    SaveImage,
    VAEDecode
)
from comfy.samplers import KSampler

# 尝试接入项目根目录下的 3D 推理器（复刻训练样张生成路径）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
  sys.path.append(PROJECT_ROOT)
try:
  from sd3_3d_lora_inference import Sd3Lora3DInferencer
  HAS_3D_INFER = True
except Exception:
  Sd3Lora3DInferencer = None
  HAS_3D_INFER = False

# GPU 选择：尊重外部传入（默认0）；设置可见后仍使用 device 0（映射到所选GPU）
if "CUDA_VISIBLE_DEVICES" not in os.environ:
  os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU_ID", "0")
torch.cuda.set_device(0)

def main():
  """主函数：从prompt.txt读取单张生成"""
  # 检查是否存在prompt.txt
  script_dir = os.path.dirname(os.path.abspath(__file__))
  prompt_file = os.path.join(script_dir, "prompt.txt")
  
  if os.path.exists(prompt_file):
    # 从prompt.txt读取提示词，单张生成
    return main_single_from_file(prompt_file)
  
  # 如果开启3D融合且推理器可用，则走与训练一致的3D+LoRA路径
  enable_3d = os.environ.get("ENABLE_3D_FUSION", "1") == "1"
  if enable_3d and HAS_3D_INFER:
    return main_3d()
  # 否则回退到原有Comfy工作流
  return main_comfy()


def main_single_from_file(prompt_file: str):
  """从prompt.txt读取提示词，单张生成"""
  # 读取提示词
  with open(prompt_file, 'r', encoding='utf-8') as f:
    prompt = f.readline().strip()
  
  if not prompt:
    print("错误: prompt.txt 文件为空")
    return
  
  print(f"读取提示词: {prompt}")
  
  # 切到项目根，便于相对路径解析（NPZ/权重等）
  os.chdir(PROJECT_ROOT)

  # 基础路径（按项目结构）
  pretrained_model = "./Stable-diffusion/SD3/sd3.5_medium.safetensors"
  lora_weights = os.environ.get("LORA_WEIGHTS", "./output/sd35m-test-lora-1e-4-000100.safetensors")
  clip_l = "./clip/clip_l.safetensors"
  clip_g = "./clip/clip_g.safetensors"
  t5xxl = "./clip/t5xxl_fp16.safetensors"
  pointcloud_dir = os.environ.get("POINTCLOUD_DIR", "./3d_features")

  # 点云路径（从环境变量或默认）
  pointcloud_path = os.environ.get("PC_PATH", os.path.join(pointcloud_dir, "44.pt"))

  # 输出目录
  out_dir = os.path.join(PROJECT_ROOT, "sd3_lora_generation_code", "output")
  os.makedirs(out_dir, exist_ok=True)
  
  # 输出文件名（使用时间戳）
  import time
  timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
  out_path = os.path.join(out_dir, f"output_{timestamp}.png")

  # 兼容加载训练好的融合网络权重：
  # 1) 若设置了 FUSION_WEIGHTS_LOAD 则直接使用
  # 2) 否则尝试根据 LORA_WEIGHTS 同前缀推导 *_fusion_weights.safetensors
  fusion_env = os.environ.get("FUSION_WEIGHTS_LOAD")
  if not fusion_env:
    def derive_fusion_path(lora_path: str) -> str:
      if not lora_path:
        return ""
      if lora_path.endswith(".safetensors"):
        return lora_path.replace(".safetensors", "_fusion_weights.safetensors")
      return ""

    # 优先使用与 LORA_WEIGHTS 同前缀的融合权重
    candidate = derive_fusion_path(lora_weights)
    if candidate and os.path.exists(candidate):
      os.environ["FUSION_WEIGHTS_LOAD"] = candidate
      print(f"Using derived fusion weights: {candidate}")
    else:
      # 其次尝试无步数后缀的基础权重 ./output/sd35m-test-lora-1e-4_fusion_weights.safetensors
      base_try = derive_fusion_path("./output/sd35m-test-lora-1e-4.safetensors")
      if base_try and os.path.exists(base_try):
        os.environ["FUSION_WEIGHTS_LOAD"] = base_try
        print(f"Using base fusion weights: {base_try}")
      else:
        print("Warning: fusion weights not found; proceeding without explicit FUSION_WEIGHTS_LOAD")

  fusion_method = os.environ.get("FUSION_METHOD", "hybrid_film_attn")
  use_t5_only = os.environ.get("USE_T5_ONLY", "1") == "1"
  shift = float(os.environ.get("DISCRETE_FLOW_SHIFT", "3.185"))

  infer = Sd3Lora3DInferencer(
    pretrained_model=pretrained_model,
    lora_weights=lora_weights,
    clip_l=clip_l,
    clip_g=clip_g,
    t5xxl=t5xxl,
    vae=None,
    device="cuda",
    dtype_str="bf16",
    enable_3d_fusion=True,
    pointcloud_dir=pointcloud_dir,
    fusion_method=fusion_method,
    pointcloud_dim=64,
    image_feature_dim=16,
    max_token_length=225,
    network_multiplier=1.0,
    use_t5xxl_cache_only=use_t5_only,
    discrete_flow_shift=shift,
  )

  infer.generate_one(
    prompt=prompt,
    negative_prompt=os.environ.get("NEGATIVE_PROMPT", "Real image"),
    height=int(os.environ.get("HEIGHT", "512")),
    width=int(os.environ.get("WIDTH", "512")),
    steps=int(os.environ.get("STEPS", "30")),
    guidance_scale=float(os.environ.get("GUIDANCE_SCALE", "7.5")),
    seed=None,
    pointcloud_path=pointcloud_path if os.path.exists(pointcloud_path) else None,
    output_path=out_path,
  )
  print(f"生成完成: {out_path}")


def main_3d():
  """三模态融合推理（使用与训练样张相同的采样与编码路径）"""
  # 切到项目根，便于相对路径解析（NPZ/权重等）
  os.chdir(PROJECT_ROOT)

  # 基础路径（按项目结构）
  pretrained_model = "./Stable-diffusion/SD3/sd3.5_medium.safetensors"
  lora_weights = os.environ.get("LORA_WEIGHTS", "./output/sd35m-test-lora-1e-4-000100.safetensors")
  clip_l = "./clip/clip_l.safetensors"
  clip_g = "./clip/clip_g.safetensors"
  t5xxl = "./clip/t5xxl_fp16.safetensors"
  pointcloud_dir = os.environ.get("POINTCLOUD_DIR", "./3d_features")

  # 提示词与点云（示例：与本脚本中默认prompt一致，如需批量可改为读 ./toml/qinglong.txt）
  prompt = "SAR image,Pilatus PC 12,Ku-band,High,VV-Polarization,0.6"
  pointcloud_path = os.environ.get("PC_PATH", os.path.join(pointcloud_dir, "44.pt"))

  # 输出
  out_dir = os.path.join(PROJECT_ROOT, "sd3_lora_generation_code", "output")
  os.makedirs(out_dir, exist_ok=True)
  out_path = os.path.join(out_dir, "output_3dfusion.png")

  # 兼容加载训练好的融合网络权重：
  # 1) 若设置了 FUSION_WEIGHTS_LOAD 则直接使用
  # 2) 否则尝试根据 LORA_WEIGHTS 同前缀推导 *_fusion_weights.safetensors
  fusion_env = os.environ.get("FUSION_WEIGHTS_LOAD")
  if not fusion_env:
    def derive_fusion_path(lora_path: str) -> str:
      if not lora_path:
        return ""
      if lora_path.endswith(".safetensors"):
        return lora_path.replace(".safetensors", "_fusion_weights.safetensors")
      return ""

    # 优先使用与 LORA_WEIGHTS 同前缀的融合权重
    candidate = derive_fusion_path(lora_weights)
    if candidate and os.path.exists(candidate):
      os.environ["FUSION_WEIGHTS_LOAD"] = candidate
      print(f"Using derived fusion weights: {candidate}")
    else:
      # 其次尝试无步数后缀的基础权重 ./output/sd35m-test-lora-1e-4_fusion_weights.safetensors
      base_try = derive_fusion_path("./output/sd35m-test-lora-1e-4.safetensors")
      if base_try and os.path.exists(base_try):
        os.environ["FUSION_WEIGHTS_LOAD"] = base_try
        print(f"Using base fusion weights: {base_try}")
      else:
        print("Warning: fusion weights not found; proceeding without explicit FUSION_WEIGHTS_LOAD")

  fusion_method = os.environ.get("FUSION_METHOD", "hybrid_film_attn")
  use_t5_only = os.environ.get("USE_T5_ONLY", "1") == "1"
  shift = float(os.environ.get("DISCRETE_FLOW_SHIFT", "3.185"))

  infer = Sd3Lora3DInferencer(
    pretrained_model=pretrained_model,
    lora_weights=lora_weights,
    clip_l=clip_l,
    clip_g=clip_g,
    t5xxl=t5xxl,
    vae=None,
    device="cuda",
    dtype_str="bf16",
    enable_3d_fusion=True,
    pointcloud_dir=pointcloud_dir,
    fusion_method=fusion_method,
    pointcloud_dim=64,
    image_feature_dim=16,
    max_token_length=225,
    network_multiplier=1.0,
    use_t5xxl_cache_only=use_t5_only,
    discrete_flow_shift=shift,
  )

  infer.generate_one(
    prompt=prompt,
    negative_prompt="",
    height=512,
    width=512,
    steps=30,
    guidance_scale=7.5,
    seed=None,
    pointcloud_path=pointcloud_path,
    output_path=out_path,
  )
  print(f"3D融合推理完成: {out_path}")


def main_comfy():
  """原 Comfy 工作流：文本图生成（回退路径）"""
  # 设置工作目录
  os.chdir(os.path.dirname(os.path.abspath(__file__)))
  
  # 初始化模型管理
  model_management.soft_empty_cache()
  
  # 1. 加载基础模型
  model, _, vae = load_base_model()
  
  # 2. 加载 LoRA
  model = load_lora(model)
  
  # 3. 设置模型采样
  model = setup_model_sampling(model)
  
  # 4. 加载 CLIP 模型
  clip_pos, clip_neg = load_clip()
  
  # 5. 创建初始潜空间
  latent = create_initial_latent()
  
  # 6. 处理正提示词条件
  positive_conditioning = process_positive_conditioning(clip_pos)
  
  # 7. 处理负提示词条件
  negative_conditioning = process_negative_conditioning(clip_neg)
  
  # 8. 执行采样
  samples = sample_images(model, latent, positive_conditioning, negative_conditioning)
  
  # 9. 解码并保存图像
  save_generated_images(samples, vae)


def main_batch_comfy():
  """在本工作流基础上进行批量生成：读取 aircraft 下的 <id>.txt 首行提示词"""
  # 工作目录指向当前脚本所在目录（Comfy 资源路径依赖）
  os.chdir(os.path.dirname(os.path.abspath(__file__)))

  # 初始化与加载（仅一次）
  model_management.soft_empty_cache()
  model, _, vae = load_base_model()
  model = load_lora(model)
  model = setup_model_sampling(model)
  clip_pos, clip_neg = load_clip()

  # 输出目录（默认 1021/，可用 OUTPUT_DIR 覆盖）
  project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  out_dir = os.environ.get("OUTPUT_DIR", os.path.join(project_root, "1"))
  os.makedirs(out_dir, exist_ok=True)

  # 采样器（与原单次一致）
  device = torch.device("cuda:0")
  sampler = KSampler(
    model,
    steps=100,
    device=device,
    sampler="dpmpp_2m",
    scheduler="ddim_uniform",
    denoise=1.0
  )

  # VAEDecode 节点
  vae_decode = VAEDecode()

  # 数据源
  train_dir = os.environ.get("TRAIN_PROMPTS_DIR", os.path.join(project_root, "train/data/train/aircraft"))
  # 可选固定负面
  negative_text = os.environ.get("NEGATIVE_PROMPT", "Real image")

  # 遍历所有 .txt（数字命名）
  txt_files = [f for f in os.listdir(train_dir) if f.endswith('.txt')]
  txt_files.sort()
  for fname in txt_files:
    sample_id = os.path.splitext(fname)[0]
    # 读取首行提示
    with open(os.path.join(train_dir, fname), 'r', encoding='utf-8') as f:
      line = f.readline().strip()
    if not line:
      continue

    # 条件组装
    positive = CLIPTextEncode().encode(clip_pos, line)[0]
    negative = CLIPTextEncode().encode(clip_neg, negative_text)[0]

    # 初始潜空间（保持 512x512）
    latent = EmptySD3LatentImage().generate(512, 512, 1)[0]
    noise = torch.randn_like(latent["samples"])  # true-random

    # 采样
    samples = sampler.sample(
      noise=noise,
      positive=positive,
      negative=negative,
      cfg=5.0,
      latent_image=latent["samples"],
      seed=None,
    )

    # 解码
    images = vae_decode.decode(vae, {"samples": samples})[0]

    # 保存（转为 PIL）
    img_tensor = images[0].detach().cpu().clamp(0, 1)
    img = tensor_to_pil(img_tensor)
    out_path = os.path.join(out_dir, f"inference_{sample_id}.png")
    img.save(out_path)
    print(f"saved -> {out_path}")


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
  # img_tensor: [C,H,W] float in [0,1]
  np_img = (img_tensor.numpy() * 255.0).astype('uint8')
  if np_img.shape[0] == 3:
    np_img = np_img.transpose(1, 2, 0)
  return Image.fromarray(np_img)

def load_base_model():
    """加载基础 SD3 模型"""
    result = comfy.sd.load_checkpoint_guess_config(
        folder_paths.get_full_path_or_raise("checkpoints", "sd3.5_medium.safetensors"),
        output_vae=True,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings")
    )
    return result[0], result[1], result[2]

def load_lora(model):
    """加载并应用 LoRA 到模型"""
    env_lora = os.environ.get("LORA_WEIGHTS")
    if env_lora and os.path.exists(env_lora):
        lora = comfy.utils.load_torch_file(env_lora, safe_load=True)
    else:
        lora_path = folder_paths.get_full_path_or_raise("loras", "sd35m-test-lora-1e-4.safetensors")
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    loaded = comfy.lora.load_lora(lora, key_map)
    model = model.clone()
    model.add_patches(loaded, 2.16)  # 修正为 2.16
    return model

def setup_model_sampling(model):
    """设置模型采样参数"""
    sampling = ModelSamplingSD3()
    shift = float(os.environ.get("DISCRETE_FLOW_SHIFT", "3.0"))
    return sampling.patch(model, shift=shift)[0]

def load_clip():
    """加载 CLIP 模型，返回用于正负提示词的两个CLIP输出"""
    clip_loader = TripleCLIPLoader()
    clips = clip_loader.load_clip(
        "clip_g.safetensors",
        "clip_l.safetensors",
        "t5xxl_fp16_2.safetensors"
    )
    return clips[0], clips[0]

def create_initial_latent():
    """创建初始潜空间，使用 SD3 版本的空白潜空间生成器"""
    latent_creator = EmptySD3LatentImage()  # 使用 SD3 版本
    return latent_creator.generate(512, 512, 1)[0]  # 尺寸为 512x512

def process_positive_conditioning(clip):
    """处理正提示词条件"""
    text_encoder = CLIPTextEncode()
    prompt = "SAR image,Cargo plane,X-band,Low"
    # prompt = "SAR image,t504,Ku-band,High,VV-Polarization"
    # prompt = "SAR image,King Air 350i,Ku-band,High,HV-Polarization"
    # prompt = "SAR image,Warship 1,C-band,High"
    # prompt = "SAR image,King Air 350i,Ku-band,High,HV-Polarization,120.0"
    # prompt = "SAR image,Military Vehicle,Ku-band,High"
    # prompt = "SAR image,Pilatus PC 12,Ku-band,High,VV-Polarization,0.6"
    positive = text_encoder.encode(clip, prompt)[0]
    return positive

def process_negative_conditioning(clip):
    """处理负提示词条件"""
    text_encoder = CLIPTextEncode()
    # prompt = "Real image,high resolution,high"
    # prompt = "Real image,low resolution,low"
    prompt = "Real image"
    negative = text_encoder.encode(clip, prompt)[0]
    return negative

def sample_images(model, latent, positive, negative):
    """执行图像采样"""
    device = torch.device("cuda:0")
    steps = int(os.environ.get("STEPS", "100"))
    sampler = KSampler(
        model,
        steps=steps,
        device=device,
        sampler="dpmpp_2m",
        scheduler="ddim_uniform",  # 修改为 ddim_uniform
        denoise=1.0
    )
    
    # 生成噪声
    noise = torch.randn_like(latent["samples"])
    
    # 可通过环境变量覆盖种子
    seed_env = os.environ.get("SEED")
    seed = int(seed_env) if (seed_env and seed_env.isdigit()) else 944396944532550
    
    cfg = float(os.environ.get("CFG", "5.0"))
    return sampler.sample(
        noise=noise,
        positive=positive,
        negative=negative,
        cfg=cfg,
        latent_image=latent["samples"],
        seed=seed
    )

def save_generated_images(samples, vae):
    """解码并保存图像"""
    # 使用 VAEDecode 解码潜空间
    vae_decode = VAEDecode()
    # 连接 KSampler 的 samples 输出和 CheckpointLoader 的 VAE 输出
    images = vae_decode.decode(vae, {"samples": samples})[0]
    
    # 保存生成的图像
    save_image = SaveImage()
    save_image.save_images(images.detach(), "output")

if __name__ == "__main__":
    main() 