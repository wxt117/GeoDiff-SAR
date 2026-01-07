#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SD3训练策略 - 支持3D点云特征融合
基于原始strategy_sd3.py，添加3D点云特征融合功能
"""

import os
import glob
import random
from typing import Any, List, Optional, Tuple, Union
import torch
import numpy as np
from transformers import CLIPTokenizer, T5TokenizerFast, CLIPTextModel, CLIPTextModelWithProjection, T5EncoderModel

from library import sd3_utils, train_util
from library import sd3_models
from library.strategy_base import LatentsCachingStrategy, TextEncodingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy

from library.utils import setup_logging

# 导入三模态特征融合模块
import sys
import importlib.util
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 动态导入模块（因为模块名以数字开头）
spec = importlib.util.spec_from_file_location(
    "three_modal_fusion", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "3d_feature_fusion_v2.py")
)
fusion_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fusion_module)
ThreeModalFusionStrategy = fusion_module.ThreeModalFusionStrategy

setup_logging()
import logging

logger = logging.getLogger(__name__)


CLIP_L_TOKENIZER_ID = "openai/clip-vit-large-patch14"
CLIP_G_TOKENIZER_ID = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
T5_XXL_TOKENIZER_ID = "google/t5-v1_1-xxl"


class Sd3TokenizeStrategy(TokenizeStrategy):
    def __init__(self, t5xxl_max_length: int = 256, tokenizer_cache_dir: Optional[str] = None) -> None:
        self.t5xxl_max_length = t5xxl_max_length
        self.clip_l = self._load_tokenizer(CLIPTokenizer, CLIP_L_TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)
        self.clip_g = self._load_tokenizer(CLIPTokenizer, CLIP_G_TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)
        self.t5xxl = self._load_tokenizer(T5TokenizerFast, T5_XXL_TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)
        self.clip_g.pad_token_id = 0  # use 0 as pad token for clip_g

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text

        l_tokens = self.clip_l(text, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
        g_tokens = self.clip_g(text, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
        t5_tokens = self.t5xxl(text, max_length=self.t5xxl_max_length, padding="max_length", truncation=True, return_tensors="pt")

        l_attn_mask = l_tokens["attention_mask"]
        g_attn_mask = g_tokens["attention_mask"]
        t5_attn_mask = t5_tokens["attention_mask"]
        l_tokens = l_tokens["input_ids"]
        g_tokens = g_tokens["input_ids"]
        t5_tokens = t5_tokens["input_ids"]

        return [l_tokens, g_tokens, t5_tokens, l_attn_mask, g_attn_mask, t5_attn_mask]


class Sd3TextEncodingStrategyWith3D(TextEncodingStrategy):
    def __init__(
        self,
        apply_lg_attn_mask: Optional[bool] = None,
        apply_t5_attn_mask: Optional[bool] = None,
        l_dropout_rate: float = 0.0,
        g_dropout_rate: float = 0.0,
        t5_dropout_rate: float = 0.0,
        pointcloud_dir: Optional[str] = None,
        enable_3d_fusion: bool = False,
        fusion_method: str = "concat",
        pointcloud_dim: int = 64,
        image_feature_dim: int = 32,
        use_t5_only: bool = False,
    ) -> None:
        """
        Args:
            apply_t5_attn_mask: Default value for apply_t5_attn_mask.
            pointcloud_dir: 3D点云特征目录
            enable_3d_fusion: 是否启用三模态特征融合
            fusion_method: 融合方法
            pointcloud_dim: 点云特征维度
            image_feature_dim: 图像特征维度
        """
        self.apply_lg_attn_mask = apply_lg_attn_mask
        self.apply_t5_attn_mask = apply_t5_attn_mask
        self.l_dropout_rate = l_dropout_rate
        self.g_dropout_rate = g_dropout_rate
        self.t5_dropout_rate = t5_dropout_rate
        
        # 三模态特征融合相关
        self.enable_3d_fusion = enable_3d_fusion
        self.pointcloud_dir = pointcloud_dir
        self.fusion_method = fusion_method
        self.pointcloud_dim = pointcloud_dim
        self.image_feature_dim = image_feature_dim
        self.use_t5_only = use_t5_only
        
        if self.enable_3d_fusion and self.pointcloud_dir:
            self.three_modal_fusion = ThreeModalFusionStrategy(
                pointcloud_dir=self.pointcloud_dir,
                fusion_method=self.fusion_method,
                pointcloud_dim=self.pointcloud_dim,
                image_feature_dim=self.image_feature_dim
            )
            # 可选加载外部融合权重（在ThreeModalFusionStrategy内部已处理），这里只打印确认与严格校验
            weights_load = os.environ.get("FUSION_WEIGHTS_LOAD", "")
            strict = os.environ.get("FUSION_WEIGHTS_STRICT", "0") == "1"
            if weights_load:
                logger.info(f"FUSION_WEIGHTS_LOAD={weights_load}")
            logger.info(f"Three-modal fusion enabled with method: {fusion_method}")
            # 严格模式：要求融合权重已成功加载到 fusion 子模块
            if strict:
                try:
                    fusion = getattr(self.three_modal_fusion, 'fusion', None)
                    if fusion is None or len(list(fusion.state_dict().keys())) == 0:
                        raise RuntimeError("fusion module is empty")
                except Exception as e:
                    raise RuntimeError(f"FUSION_WEIGHTS_STRICT: fusion weights not loaded or invalid: {e}")
        else:
            self.three_modal_fusion = None

    def encode_tokens(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: List[torch.Tensor],
        apply_lg_attn_mask: Optional[bool] = False,
        apply_t5_attn_mask: Optional[bool] = False,
        enable_dropout: bool = True,
        image_paths: Optional[List[str]] = None,
    ) -> List[torch.Tensor]:
        """
        returned embeddings are not masked
        """
        clip_l, clip_g, t5xxl = models
        clip_l: Optional[CLIPTextModel]
        clip_g: Optional[CLIPTextModelWithProjection]
        t5xxl: Optional[T5EncoderModel]

        if apply_lg_attn_mask is None:
            apply_lg_attn_mask = self.apply_lg_attn_mask
        if apply_t5_attn_mask is None:
            apply_t5_attn_mask = self.apply_t5_attn_mask

        l_tokens, g_tokens, t5_tokens, l_attn_mask, g_attn_mask, t5_attn_mask = tokens

        # dropout: if enable_dropout is False, dropout is not applied. dropout means zeroing out embeddings

        if l_tokens is None or clip_l is None:
            lg_out = None
            lg_pooled = None
            l_attn_mask = None
            g_attn_mask = None
        else:
            assert g_tokens is not None, "g_tokens must not be None if l_tokens is not None"

            # drop some members of the batch: we do not call clip_l and clip_g for dropped members
            batch_size, l_seq_len = l_tokens.shape
            g_seq_len = g_tokens.shape[1]

            non_drop_l_indices = []
            non_drop_g_indices = []
            for i in range(l_tokens.shape[0]):
                drop_l = enable_dropout and (self.l_dropout_rate > 0.0 and random.random() < self.l_dropout_rate)
                drop_g = enable_dropout and (self.g_dropout_rate > 0.0 and random.random() < self.g_dropout_rate)
                if not drop_l:
                    non_drop_l_indices.append(i)
                if not drop_g:
                    non_drop_g_indices.append(i)

            # filter out dropped members
            if len(non_drop_l_indices) > 0 and len(non_drop_l_indices) < batch_size:
                l_tokens = l_tokens[non_drop_l_indices]
                l_attn_mask = l_attn_mask[non_drop_l_indices]
            if len(non_drop_g_indices) > 0 and len(non_drop_g_indices) < batch_size:
                g_tokens = g_tokens[non_drop_g_indices]
                g_attn_mask = g_attn_mask[non_drop_g_indices]

            # call clip_l for non-dropped members
            if len(non_drop_l_indices) > 0:
                nd_l_attn_mask = l_attn_mask.to(clip_l.device)
                prompt_embeds = clip_l(
                    l_tokens.to(clip_l.device), nd_l_attn_mask if apply_lg_attn_mask else None, output_hidden_states=True
                )
                nd_l_pooled = prompt_embeds[0]
                nd_l_out = prompt_embeds.hidden_states[-2]
            if len(non_drop_g_indices) > 0:
                nd_g_attn_mask = g_attn_mask.to(clip_g.device)
                prompt_embeds = clip_g(
                    g_tokens.to(clip_g.device), nd_g_attn_mask if apply_lg_attn_mask else None, output_hidden_states=True
                )
                nd_g_pooled = prompt_embeds[0]
                nd_g_out = prompt_embeds.hidden_states[-2]

            # fill in the dropped members
            if len(non_drop_l_indices) == batch_size:
                l_pooled = nd_l_pooled
                l_out = nd_l_out
            else:
                # model output is always float32 because of the models are wrapped with Accelerator
                l_pooled = torch.zeros((batch_size, 768), device=clip_l.device, dtype=torch.float32)
                l_out = torch.zeros((batch_size, l_seq_len, 768), device=clip_l.device, dtype=torch.float32)
                l_attn_mask = torch.zeros((batch_size, l_seq_len), device=clip_l.device, dtype=l_attn_mask.dtype)
                if len(non_drop_l_indices) > 0:
                    l_pooled[non_drop_l_indices] = nd_l_pooled
                    l_out[non_drop_l_indices] = nd_l_out
                    l_attn_mask[non_drop_l_indices] = nd_l_attn_mask

            if len(non_drop_g_indices) == batch_size:
                g_pooled = nd_g_pooled
                g_out = nd_g_out
            else:
                g_pooled = torch.zeros((batch_size, 1280), device=clip_g.device, dtype=torch.float32)
                g_out = torch.zeros((batch_size, g_seq_len, 1280), device=clip_g.device, dtype=torch.float32)
                g_attn_mask = torch.zeros((batch_size, g_seq_len), device=clip_g.device, dtype=g_attn_mask.dtype)
                if len(non_drop_g_indices) > 0:
                    g_pooled[non_drop_g_indices] = nd_g_pooled
                    g_out[non_drop_g_indices] = nd_g_out
                    g_attn_mask[non_drop_g_indices] = nd_g_attn_mask

            lg_pooled = torch.cat((l_pooled, g_pooled), dim=-1)
            lg_out = torch.cat([l_out, g_out], dim=-1)

        if t5xxl is None or t5_tokens is None:
            t5_out = None
            t5_attn_mask = None
        else:
            # drop some members of the batch: we do not call t5xxl for dropped members
            batch_size, t5_seq_len = t5_tokens.shape
            non_drop_t5_indices = []
            for i in range(t5_tokens.shape[0]):
                drop_t5 = enable_dropout and (self.t5_dropout_rate > 0.0 and random.random() < self.t5_dropout_rate)
                if not drop_t5:
                    non_drop_t5_indices.append(i)

            # filter out dropped members
            if len(non_drop_t5_indices) > 0 and len(non_drop_t5_indices) < batch_size:
                t5_tokens = t5_tokens[non_drop_t5_indices]
                t5_attn_mask = t5_attn_mask[non_drop_t5_indices]

            # call t5xxl for non-dropped members
            if len(non_drop_t5_indices) > 0:
                nd_t5_attn_mask = t5_attn_mask.to(t5xxl.device)
                nd_t5_out, _ = t5xxl(
                    t5_tokens.to(t5xxl.device),
                    nd_t5_attn_mask if apply_t5_attn_mask else None,
                    return_dict=False,
                    output_hidden_states=True,
                )

            # fill in the dropped members
            if len(non_drop_t5_indices) == batch_size:
                t5_out = nd_t5_out
            else:
                t5_out = torch.zeros((batch_size, t5_seq_len, 4096), device=t5xxl.device, dtype=torch.float32)
                t5_attn_mask = torch.zeros((batch_size, t5_seq_len), device=t5xxl.device, dtype=t5_attn_mask.dtype)
                if len(non_drop_t5_indices) > 0:
                    t5_out[non_drop_t5_indices] = nd_t5_out
                    t5_attn_mask[non_drop_t5_indices] = nd_t5_attn_mask

        # 三模态特征融合
        if self.enable_3d_fusion and self.three_modal_fusion is not None and image_paths is not None:
            try:
                # 融合文本、点云和图像特征
                # 确保数据类型匹配
                original_dtype = lg_pooled.dtype
                if lg_pooled.dtype != torch.float32:
                    lg_pooled_float = lg_pooled.float()
                else:
                    lg_pooled_float = lg_pooled
                
                fused_lg_pooled = self.three_modal_fusion.process_batch(lg_pooled_float, image_paths)
                
                # 转换回原始数据类型
                if fused_lg_pooled.dtype != original_dtype:
                    fused_lg_pooled = fused_lg_pooled.to(original_dtype)
                
                logger.debug(f"Three-modal fusion applied: {lg_pooled.shape} -> {fused_lg_pooled.shape}")
                lg_pooled = fused_lg_pooled
            except Exception as e:
                logger.warning(f"Three-modal fusion failed: {e}, using original features")

        # 仅用T5（训练use_t5xxl_cache_only一致）：将序列级CLIP输出置零，但保留融合后的 pooled 作为条件
        if self.use_t5_only:
            if lg_out is not None:
                lg_out = torch.zeros_like(lg_out)

        # masks are used for attention masking in transformer
        return [lg_out, t5_out, lg_pooled, l_attn_mask, g_attn_mask, t5_attn_mask]

    def drop_cached_text_encoder_outputs(
        self,
        lg_out: torch.Tensor,
        t5_out: torch.Tensor,
        lg_pooled: torch.Tensor,
        l_attn_mask: torch.Tensor,
        g_attn_mask: torch.Tensor,
        t5_attn_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # dropout: if enable_dropout is True, dropout is not applied. dropout means zeroing out embeddings
        if lg_out is not None:
            for i in range(lg_out.shape[0]):
                drop_l = self.l_dropout_rate > 0.0 and random.random() < self.l_dropout_rate
                if drop_l:
                    lg_out[i, :, :768] = torch.zeros_like(lg_out[i, :, :768])
                    lg_pooled[i, :768] = torch.zeros_like(lg_pooled[i, :768])
                    if l_attn_mask is not None:
                        l_attn_mask[i] = torch.zeros_like(l_attn_mask[i])
                drop_g = self.g_dropout_rate > 0.0 and random.random() < self.g_dropout_rate
                if drop_g:
                    lg_out[i, :, 768:] = torch.zeros_like(lg_out[i, :, 768:])
                    lg_pooled[i, 768:] = torch.zeros_like(lg_pooled[i, 768:])
                    if g_attn_mask is not None:
                        g_attn_mask[i] = torch.zeros_like(g_attn_mask[i])

        if t5_out is not None:
            for i in range(t5_out.shape[0]):
                drop_t5 = self.t5_dropout_rate > 0.0 and random.random() < self.t5_dropout_rate
                if drop_t5:
                    t5_out[i] = torch.zeros_like(t5_out[i])
                    if t5_attn_mask is not None:
                        t5_attn_mask[i] = torch.zeros_like(t5_attn_mask[i])

        return [lg_out, t5_out, lg_pooled, l_attn_mask, g_attn_mask, t5_attn_mask]

    def concat_encodings(
        self, lg_out: torch.Tensor, t5_out: Optional[torch.Tensor], lg_pooled: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        if t5_out is None:
            t5_out = torch.zeros((lg_out.shape[0], 77, 4096), device=lg_out.device, dtype=lg_out.dtype)
        return torch.cat([lg_out, t5_out], dim=-2), lg_pooled


class Sd3TextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    SD3_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_sd3_te.npz"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
        apply_lg_attn_mask: bool = False,
        apply_t5_attn_mask: bool = False,
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)
        self.apply_lg_attn_mask = apply_lg_attn_mask
        self.apply_t5_attn_mask = apply_t5_attn_mask

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + Sd3TextEncoderOutputsCachingStrategy.SD3_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str):
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            npz = np.load(npz_path)
            expected_keys = ["lg_out", "t5_out", "lg_pooled", "l_attn_mask", "g_attn_mask", "t5_attn_mask"]
            return all(key in npz for key in expected_keys)
        except Exception:
            return False

    def load_disk_cached_outputs(self, npz_path: str) -> List[torch.Tensor]:
        npz = np.load(npz_path)
        return [
            torch.from_numpy(npz["lg_out"]),
            torch.from_numpy(npz["t5_out"]),
            torch.from_numpy(npz["lg_pooled"]),
            torch.from_numpy(npz["l_attn_mask"]),
            torch.from_numpy(npz["g_attn_mask"]),
            torch.from_numpy(npz["t5_attn_mask"]),
        ]

    def save_disk_cached_outputs(self, npz_path: str, outputs: List[torch.Tensor]):
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        np.savez(
            npz_path,
            lg_out=outputs[0].cpu().numpy(),
            t5_out=outputs[1].cpu().numpy(),
            lg_pooled=outputs[2].cpu().numpy(),
            l_attn_mask=outputs[3].cpu().numpy(),
            g_attn_mask=outputs[4].cpu().numpy(),
            t5_attn_mask=outputs[5].cpu().numpy(),
        )

    def get_text_encoder_outputs_caching_strategy(self, args):
        return Sd3TextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk,
            args.text_encoder_batch_size,
            args.skip_cache_check,
            is_partial=args.cache_text_encoder_outputs,
            apply_lg_attn_mask=args.apply_lg_attn_mask,
            apply_t5_attn_mask=args.apply_t5_attn_mask,
        )
    
    def encode_tokens_with_weights(
        self,
        tokenize_strategy: "TokenizeStrategy",
        models: List[Any],
        input_ids_list: List[torch.Tensor],
        weights_list: List[torch.Tensor],
        image_paths: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        带权重的文本编码 - 支持3D融合
        """
        # 调用基类的带权重编码方法
        lg_out, t5_out, lg_pooled, l_attn_mask, g_attn_mask, t5_attn_mask = super().encode_tokens_with_weights(
            tokenize_strategy, models, input_ids_list, weights_list
        )
        
        # 三模态特征融合
        if self.enable_3d_fusion and self.three_modal_fusion is not None and image_paths is not None:
            try:
                # 融合文本、点云和图像特征
                # 确保数据类型匹配
                original_dtype = lg_pooled.dtype
                if lg_pooled.dtype != torch.float32:
                    lg_pooled_float = lg_pooled.float()
                else:
                    lg_pooled_float = lg_pooled
                
                fused_lg_pooled = self.three_modal_fusion.process_batch(lg_pooled_float, image_paths)
                
                # 转换回原始数据类型
                if fused_lg_pooled.dtype != original_dtype:
                    fused_lg_pooled = fused_lg_pooled.to(original_dtype)
                
                logger.debug(f"Three-modal fusion applied (with weights): {lg_pooled.shape} -> {fused_lg_pooled.shape}")
                lg_pooled = fused_lg_pooled
            except Exception as e:
                logger.warning(f"Three-modal fusion failed (with weights): {e}, using original features")
        
        return lg_out, t5_out, lg_pooled, l_attn_mask, g_attn_mask, t5_attn_mask