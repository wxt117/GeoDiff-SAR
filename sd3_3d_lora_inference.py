#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复刻训练样张生成的 SD3.5 + LoRA + 三模态融合 推理脚本
核心对齐点：
- 文本编码与三模态融合：library/strategy_sd3_with_3d.Sd3TextEncodingStrategyWith3D
- 采样：library/sd3_train_utils.do_sample（与训练样张一致的采样路径）
- LoRA 注入：networks/lora_sd3.create_network_from_weights + apply_to + load_weights + set_multiplier
"""

import argparse
import os
import time
import glob
from typing import Optional, Tuple, List, Dict, Any

import torch
import numpy as np
from PIL import Image
from safetensors.torch import load_file

import sys
# 添加 sd-scripts 到路径（使用绝对路径以确保从任何目录运行都能找到）
_script_dir = os.path.dirname(os.path.abspath(__file__))
_sd_scripts_path = os.path.join(_script_dir, 'sd-scripts')
if os.path.exists(_sd_scripts_path) and _sd_scripts_path not in sys.path:
    sys.path.insert(0, _sd_scripts_path)
from library import sd3_utils, train_util
from library.device_utils import clean_memory_on_device
from contextlib import nullcontext

from library.strategy_sd3_with_3d import (
  Sd3TokenizeStrategy,
  Sd3TextEncodingStrategyWith3D,
)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_dtype(dtype_str: str) -> torch.dtype:
  mapping = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
  }
  return mapping.get(dtype_str, torch.bfloat16)


class Sd3Lora3DInferencer:
  def __init__(
    self,
    pretrained_model: str,
    lora_weights: Optional[str],
    clip_l: str,
    clip_g: str,
    t5xxl: str,
    vae: Optional[str],
    device: str,
    dtype_str: str,
    enable_3d_fusion: bool,
    pointcloud_dir: Optional[str],
    fusion_method: str,
    pointcloud_dim: int,
    image_feature_dim: int,
    max_token_length: int,
    network_multiplier: float,
    use_t5xxl_cache_only: bool,
    discrete_flow_shift: float = 3.185,
  ) -> None:
    self.device = torch.device(device)
    self.weight_dtype = get_dtype(dtype_str)

    # 1) 加载基础模型（统一 checkpoint）
    logger.info(f"加载SD3模型: {pretrained_model}")
    state_dict = load_file(pretrained_model)
    self.mmdit = sd3_utils.load_mmdit(state_dict, self.weight_dtype, self.device)
    self.mmdit = self.mmdit.to(self.device)
    self.mmdit.eval()

    # 2) 加载 VAE（从独立路径或从统一 ckpt 里）
    if vae:
      _ = load_file(vae)  # 仅验证文件存在与可读
      self.vae = sd3_utils.load_vae(vae, self.weight_dtype, 'cpu')
    else:
      self.vae = sd3_utils.load_vae(None, self.weight_dtype, 'cpu', state_dict=state_dict)
    self.vae.eval()

    # 3) 加载文本编码器（CLIP-L, CLIP-G, T5）
    self.clip_l = sd3_utils.load_clip_l(clip_l, self.weight_dtype, self.device)
    self.clip_l = self.clip_l.to(self.device)
    self.clip_l.eval()

    self.clip_g = sd3_utils.load_clip_g(clip_g, self.weight_dtype, self.device)
    self.clip_g = self.clip_g.to(self.device)
    self.clip_g.eval()

    # T5 加载到与其他模块相同的 dtype/device（训练中为 bf16+cuda；此处跟随全局 dtype）
    self.t5xxl = sd3_utils.load_t5xxl(t5xxl, self.weight_dtype, self.device)
    self.t5xxl = self.t5xxl.to(self.device)
    self.t5xxl.eval()

    self.text_encoders = [self.clip_l, self.clip_g, self.t5xxl]
    self.use_t5_only = use_t5xxl_cache_only
    self.discrete_flow_shift = discrete_flow_shift

    # 先初始化分词与编码/融合策略，之后再尝试加载训练导出的融合配置
    self._init_encoding_strategy(
      enable_3d_fusion=enable_3d_fusion,
      pointcloud_dir=pointcloud_dir,
      fusion_method=fusion_method,
      pointcloud_dim=pointcloud_dim,
      image_feature_dim=image_feature_dim,
      max_token_length=max_token_length,
    )

    # 如果存在训练导出的融合配置，自动加载并应用（可能会覆盖入参）
    try:
      import json
      project_root = os.path.dirname(os.path.abspath(__file__))
      # 1) 优先尝试与 LoRA 同目录同名（.safetensors -> _fusion.json）
      fusion_json = None
      if lora_weights:
        lora_dir = os.path.dirname(os.path.abspath(lora_weights))
        lora_base = os.path.splitext(os.path.basename(lora_weights))[0]
        cand = os.path.join(lora_dir, f"{lora_base}_fusion.json")
        if os.path.exists(cand):
          fusion_json = cand
      # 2) 其次尝试项目 output 目录下最新的 *_fusion.json
      if fusion_json is None:
        out_dir = os.path.join(project_root, 'output')
        if os.path.isdir(out_dir):
          import glob
          files = glob.glob(os.path.join(out_dir, "*_fusion.json"))
          if files:
            fusion_json = max(files, key=os.path.getmtime)
      if fusion_json and os.path.exists(fusion_json):
        with open(fusion_json, 'r', encoding='utf-8') as f:
          cfg = json.load(f)
        # 应用关键项
        if cfg.get('fusion_method'):
          fusion_method = cfg['fusion_method']
        if cfg.get('pointcloud_dim'):
          pointcloud_dim = int(cfg['pointcloud_dim'])
        if cfg.get('image_feature_dim'):
          image_feature_dim = int(cfg['image_feature_dim'])
        if cfg.get('enable_3d_fusion') is not None:
          enable_3d_fusion = bool(cfg['enable_3d_fusion'])
        if cfg.get('fusion_seed') is not None:
          try:
            import random
            seedv = int(cfg['fusion_seed'])
            random.seed(seedv); np.random.seed(seedv); torch.manual_seed(seedv)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seedv)
          except Exception:
            pass
        if cfg.get('gentle_alpha') is not None:
          os.environ['GENTLE_ALPHA'] = str(cfg['gentle_alpha'])
        if cfg.get('use_t5xxl_cache_only') is not None:
          self.use_t5_only = bool(cfg['use_t5xxl_cache_only'])
        logger.info(f"Loaded fusion config: {fusion_json}")
        # 重新初始化编码与融合策略，确保配置立即生效
        try:
          self._init_encoding_strategy(
            enable_3d_fusion=enable_3d_fusion,
            pointcloud_dir=pointcloud_dir,
            fusion_method=fusion_method,
            pointcloud_dim=pointcloud_dim,
            image_feature_dim=image_feature_dim,
            max_token_length=max_token_length,
          )
          logger.info("Reinitialized encoding strategy with loaded fusion config")
        except Exception as _e:
          logger.warning(f"Reinit encoding strategy failed: {_e}")
    except Exception as e:
      logger.warning(f"Load fusion config failed: {e}")

    # 4) LoRA 注入（按训练路径）
    self.lora_network = None
    if lora_weights:
      self._load_lora(lora_weights, network_multiplier)

    # 编码/融合策略已在 _init_encoding_strategy 中完成初始化

  def _init_encoding_strategy(
    self,
    enable_3d_fusion: bool,
    pointcloud_dir: Optional[str],
    fusion_method: str,
    pointcloud_dim: int,
    image_feature_dim: int,
    max_token_length: int,
  ) -> None:
    """初始化分词与文本编码/三模态融合策略。"""
    self.tokenize_strategy = Sd3TokenizeStrategy(
      t5xxl_max_length=max_token_length,
      tokenizer_cache_dir=None,
    )
    self.encoding_strategy = Sd3TextEncodingStrategyWith3D(
      apply_lg_attn_mask=False,
      apply_t5_attn_mask=True,
      l_dropout_rate=0.0,
      g_dropout_rate=0.0,
      t5_dropout_rate=0.0,
      pointcloud_dir=pointcloud_dir if enable_3d_fusion else None,
      enable_3d_fusion=enable_3d_fusion,
      fusion_method=fusion_method,
      pointcloud_dim=pointcloud_dim,
      image_feature_dim=image_feature_dim,
      use_t5_only=self.use_t5_only,
    )

  def _load_lora(self, lora_path: str, multiplier: float) -> None:
    if not os.path.exists(lora_path):
      logger.warning(f"LoRA 权重不存在: {lora_path}")
      return
    import importlib.util
    lora_py = os.path.join(os.path.dirname(__file__), 'sd-scripts', 'networks', 'lora_sd3.py')
    spec = importlib.util.spec_from_file_location('lora_sd3', lora_py)
    if not spec or not spec.loader:
      raise RuntimeError(f"无法加载 LoRA 模块: {lora_py}")
    lora_sd3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lora_sd3)
    create_network_from_weights = lora_sd3.create_network_from_weights

    logger.info(f"加载 LoRA: {lora_path} (multiplier={multiplier})")
    network, _ = create_network_from_weights(
      multiplier,
      lora_path,
      self.vae,
      self.text_encoders,
      self.mmdit,
      for_inference=True,
    )
    # 按训练样张路径注入（优先 UNet；若存在 TE 模块也不妨碍，权重文件通常无 TE 模块）
    # 与训练样张一致：仅作用于 UNet（train_unet_only）
    network.apply_to(self.text_encoders, self.mmdit, apply_text_encoder=False, apply_unet=True)
    _info = network.load_weights(lora_path)
    network.set_multiplier(multiplier)
    network.to(self.device, dtype=self.mmdit.dtype)
    # 为避免 block swap 等路径弱化 LoRA，执行预合并（不改动模块结构，仅加权到原权重）
    try:
      network.pre_calculation()
      logger.info("LoRA 预合并已应用（pre_calculation）")
    except Exception as e:
      logger.warning(f"LoRA 预合并失败（忽略）：{e}")
    # 记录模块数量
    try:
      te_count = len(getattr(network, 'text_encoder_loras', []))
      unet_count = len(getattr(network, 'unet_loras', []))
      logger.info(f"LoRA 模块数: TE={te_count}, UNet={unet_count}")
    except Exception:
      pass
    self.lora_network = network
    logger.info("LoRA 注入完成")

  def _resolve_image_paths_from_pointcloud(self, pointcloud_path: Optional[str]) -> Optional[List[str]]:
    if not pointcloud_path:
      return None
    pc_filename = os.path.basename(pointcloud_path)
    sample_id = os.path.splitext(pc_filename)[0]
    # 以项目根目录为基准，或优先使用 NPZ_DIR 覆盖，增强匹配策略
    project_root = os.path.dirname(os.path.abspath(__file__))
    npz_dir = os.environ.get("NPZ_DIR", os.path.join(project_root, "train", "data", "train", "aircraft"))
    # 1) 优先匹配 <id>_*_sd3.npz（训练缓存命名）
    files = glob.glob(os.path.join(npz_dir, f"{sample_id}_*_sd3.npz"))
    # 2) 其次匹配 <id>.npz（用户提供的简化命名）
    if not files:
      files = glob.glob(os.path.join(npz_dir, f"{sample_id}.npz"))
    if files:
      return [f"{files[0]}|pc={os.path.abspath(pointcloud_path)}"]
    # 未找到NPZ时，返回占位路径，令三模态融合以零图像特征作为fallback
    return [f"missing_npz://{sample_id}|pc={os.path.abspath(pointcloud_path)}"]

  def encode_prompt(self, prompt: str, pointcloud_path: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens_and_masks = self.tokenize_strategy.tokenize(prompt)
    tokens_and_masks = [t.to(self.device) if t is not None else None for t in tokens_and_masks]
    image_paths = self._resolve_image_paths_from_pointcloud(pointcloud_path)

    lg_out, t5_out, lg_pooled, l_mask, g_mask, t5_mask = self.encoding_strategy.encode_tokens(
      self.tokenize_strategy,
      self.text_encoders,
      tokens_and_masks,
      apply_lg_attn_mask=False,
      apply_t5_attn_mask=True,
      enable_dropout=False,
      image_paths=image_paths,
    )
    # 按训练脚本开关：仅使用T5缓存时，置零CLIP输出与pooled
    if self.use_t5_only:
      if lg_out is not None:
        lg_out = torch.zeros_like(lg_out)
      else:
        # 构造与正常尺寸一致的零特征 (B, 77, 2048)
        batch_size = t5_out.shape[0] if t5_out is not None else 1
        lg_out = torch.zeros((batch_size, 77, 2048), device=self.device, dtype=t5_out.dtype if t5_out is not None else torch.float32)
      if lg_pooled is not None:
        lg_pooled = torch.zeros_like(lg_pooled)
      else:
        lg_pooled = torch.zeros((lg_out.shape[0], 768 + 1280), device=self.device, dtype=lg_out.dtype)

    # 与训练一致：拼接 CLIP 与 T5 文本编码（若 use_t5_only 则 CLIP 为零向量）
    context, pooled = self.encoding_strategy.concat_encodings(lg_out, t5_out, lg_pooled)
    # context/pooled 会在 do_sample 内部被拼接 cond/neg_cond 并转到 dtype+device
    return context, pooled

  @torch.no_grad()
  def generate_one(
    self,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    seed: Optional[int],
    pointcloud_path: Optional[str],
    output_path: Optional[str],
  ) -> Image.Image:
    # 尺寸对齐 8 的倍数
    height = max(64, height - height % 8)
    width = max(64, width - width % 8)

    # 完全走训练样张路径：调用 sample_image_inference
    from library import sd3_train_utils

    class _Accel:
      def __init__(self, device, dtype):
        self.device = device
        self._dtype = dtype
        self.trackers = []  # no trackers by default
      def autocast(self):
        if self.device.type == 'cuda' and self._dtype in (torch.bfloat16, torch.float16):
          return torch.autocast(device_type='cuda', dtype=self._dtype)
        return nullcontext()
      def get_tracker(self, name: str):
        class _Dummy:
          def __init__(self):
            self.name = name
        return _Dummy()

    # 构造最小 args
    Args = type('Args', (), {})
    args = Args()
    args.enable_3d_fusion = self.encoding_strategy.enable_3d_fusion
    args.fusion_method = getattr(self.encoding_strategy, 'fusion_method', 'attention_gated')
    args.pointcloud_dir = getattr(self.encoding_strategy, 'pointcloud_dir', None)
    args.apply_lg_attn_mask = False
    args.apply_t5_attn_mask = True
    # 由环境变量控制输出名前缀；未设置则不加任何前缀
    out_name_env = os.environ.get('OUTPUT_NAME')
    args.output_name = (out_name_env if out_name_env not in (None, '', 'None') else None)
    # 采样器离散流位移（对齐训练脚本配置）
    args.discrete_flow_shift = getattr(self, 'discrete_flow_shift', 3.185)

    prompt_dict = {
      'prompt': prompt,
      'negative_prompt': negative_prompt or '',
      'width': width,
      'height': height,
      'sample_steps': steps,
      'scale': guidance_scale,
      'seed': seed,
      'pointcloud_path': pointcloud_path,
      'enum': 0,
    }

    save_dir = os.path.dirname(output_path) if output_path else './output/inference'
    os.makedirs(save_dir, exist_ok=True)

    acc = _Accel(self.device, self.weight_dtype)
    # 设置全局策略供 sample_image_inference 查询（若未设置时再设置，避免重复设置报错）
    from library import strategy_base
    if strategy_base.TextEncodingStrategy.get_strategy() is None:
      strategy_base.TextEncodingStrategy.set_strategy(self.encoding_strategy)
    if strategy_base.TokenizeStrategy.get_strategy() is None:
      strategy_base.TokenizeStrategy.set_strategy(self.tokenize_strategy)

    sd3_train_utils.sample_image_inference(
      acc,
      args,
      self.mmdit,
      self.text_encoders,
      self.vae,
      save_dir,
      prompt_dict,
      epoch=None,
      steps=0,
      sample_prompts_te_outputs=None,
      prompt_replacement=None,
    )

    # 读取刚保存的文件（按时间戳匹配）
    latest = sorted([p for p in os.listdir(save_dir) if p.endswith('.png')])[-1]
    img_path = os.path.join(save_dir, latest)

    # 若指定了目标路径，则将该文件“搬移/改名”为最终结果，避免产生两份图片
    if output_path and img_path != output_path:
      os.makedirs(os.path.dirname(output_path), exist_ok=True)
      try:
        os.replace(img_path, output_path)
        img_path = output_path
      except Exception:
        # 回退：无法替换时再做一次保存
        Image.open(img_path).convert('RGB').save(output_path)
        img_path = output_path

    img = Image.open(img_path).convert('RGB')
    logger.info(f"保存: {img_path}")
    return img


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description='SD3.5 三模态融合 LoRA 推理（复刻训练样张生成）')
  # 模型
  p.add_argument('--pretrained_model', type=str, required=True)
  p.add_argument('--lora_weights', type=str, default=None)
  p.add_argument('--clip_l', type=str, required=True)
  p.add_argument('--clip_g', type=str, required=True)
  p.add_argument('--t5xxl', type=str, required=True)
  p.add_argument('--vae', type=str, default=None)

  # 三模态
  p.add_argument('--enable_3d_fusion', action='store_true')
  p.add_argument('--pointcloud_dir', type=str, default=None)
  p.add_argument('--fusion_method', type=str, default='attention_gated', choices=['concat','add','attention','gentle','gated','attention_gated'])
  p.add_argument('--pointcloud_dim', type=int, default=64)
  p.add_argument('--image_feature_dim', type=int, default=16)

  # 生成
  p.add_argument('--prompt', type=str, default=None)
  p.add_argument('--prompts_file', type=str, default=None)
  p.add_argument('--negative_prompt', type=str, default='')
  p.add_argument('--height', type=int, default=512)
  p.add_argument('--width', type=int, default=512)
  p.add_argument('--sample_steps', type=int, default=30)
  p.add_argument('--guidance_scale', type=float, default=7.5)
  p.add_argument('--seed', type=int, default=None)
  p.add_argument('--network_multiplier', type=float, default=1.0)
  p.add_argument('--max_token_length', type=int, default=225)
  p.add_argument('--pointcloud_path', type=str, default=None)
  p.add_argument('--use_t5xxl_cache_only', action='store_true')
  # 采样器离散流位移（Euler Discrete Flow shift），默认与训练脚本一致
  p.add_argument('--discrete_flow_shift', type=float, default=3.185)

  # 输出/设备
  p.add_argument('--output_dir', type=str, default='./output/inference')
  p.add_argument('--output_name', type=str, default=None)
  p.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'])
  p.add_argument('--dtype', type=str, default='bf16', choices=['fp32','fp16','bf16'])
  return p.parse_args()


def main() -> None:
  args = parse_args()
  if args.prompt is None and args.prompts_file is None:
    raise SystemExit('必须指定 --prompt 或 --prompts_file')
  if args.enable_3d_fusion and not args.pointcloud_dir:
    raise SystemExit('启用三模态融合时必须指定 --pointcloud_dir')

  os.makedirs(args.output_dir, exist_ok=True)

  infer = Sd3Lora3DInferencer(
    pretrained_model=args.pretrained_model,
    lora_weights=args.lora_weights,
    clip_l=args.clip_l,
    clip_g=args.clip_g,
    t5xxl=args.t5xxl,
    vae=args.vae,
    device=args.device,
    dtype_str=args.dtype,
    enable_3d_fusion=args.enable_3d_fusion,
    pointcloud_dir=args.pointcloud_dir,
    fusion_method=args.fusion_method,
    pointcloud_dim=args.pointcloud_dim,
    image_feature_dim=args.image_feature_dim,
    max_token_length=args.max_token_length,
    network_multiplier=args.network_multiplier,
    use_t5xxl_cache_only=args.use_t5xxl_cache_only,
    discrete_flow_shift=args.discrete_flow_shift,
  )

  if args.prompt is not None:
    ts = time.strftime('%Y%m%d%H%M%S', time.localtime())
    seed_suffix = '' if args.seed is None else f'_{args.seed}'
    name = args.output_name or 'generated'
    out_path = os.path.join(args.output_dir, f'{name}_{ts}{seed_suffix}.png')
    infer.generate_one(
      prompt=args.prompt,
      negative_prompt=args.negative_prompt,
      height=args.height,
      width=args.width,
      steps=args.sample_steps,
      guidance_scale=args.guidance_scale,
      seed=args.seed,
      pointcloud_path=args.pointcloud_path,
      output_path=out_path,
    )
  else:
    # 与训练一致：使用 train_util.load_prompts 解析 txt，支持行内 --pc
    prompts = train_util.load_prompts(args.prompts_file)
    logger.info(f"共 {len(prompts)} 条提示词")
    for i, p in enumerate(prompts):
      if isinstance(p, str):
        prompt = p
        negative_prompt = args.negative_prompt
        height = args.height
        width = args.width
        steps = args.sample_steps
        scale = args.guidance_scale
        seed = args.seed
        pc_path = args.pointcloud_path
      else:
        prompt = p.get('prompt', '')
        negative_prompt = p.get('negative_prompt', args.negative_prompt)
        height = p.get('height', args.height)
        width = p.get('width', args.width)
        steps = p.get('sample_steps', args.sample_steps)
        scale = p.get('scale', args.guidance_scale)
        seed = p.get('seed', args.seed)
        pc_path = p.get('pointcloud_path', args.pointcloud_path)

      ts = time.strftime('%Y%m%d%H%M%S', time.localtime())
      seed_suffix = '' if seed is None else f'_{seed}'
      name = args.output_name or 'generated'
      out_path = os.path.join(args.output_dir, f'{name}_{i:02d}_{ts}{seed_suffix}.png')
      try:
        infer.generate_one(
          prompt=prompt,
          negative_prompt=negative_prompt,
          height=height,
          width=width,
          steps=steps,
          guidance_scale=scale,
          seed=seed,
          pointcloud_path=pc_path,
          output_path=out_path,
        )
      except Exception as e:
        logger.error(f"生成失败 #{i}: {e}")
        continue

  logger.info('全部完成')


if __name__ == '__main__':
  main()


