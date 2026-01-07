#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三模态特征融合模块
将3D点云特征、图像特征与文本特征进行融合，用于SD3.5训练
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
try:
    from safetensors.torch import save_file as safetensors_save, load_file as safetensors_load
    _HAS_SAFETENSORS = True
except Exception:
    _HAS_SAFETENSORS = False

logger = logging.getLogger(__name__)


class ImageFeatureLoader:
    """图像特征加载器 - 从npz文件中提取图像潜在特征"""
    
    def __init__(self, image_feature_dim: int = 32):
        """
        初始化图像特征加载器
        
        Args:
            image_feature_dim: 图像特征维度（潜在空间维度）
        """
        self.image_feature_dim = image_feature_dim
        self.feature_cache = {}
        
    def load_image_features(self, npz_path: str) -> torch.Tensor:
        """
        从npz文件中加载图像潜在特征
        
        Args:
            npz_path: npz文件路径
            
        Returns:
            图像特征张量 [batch_size, channels, height, width]
        """
        if npz_path in self.feature_cache:
            return self.feature_cache[npz_path]
            
        try:
            # 加载npz文件
            data = np.load(npz_path)
            
            # 查找latents相关的键（可能是不同分辨率）
            latents_key = None
            for key in data.keys():
                if key.startswith('latents_'):
                    latents_key = key
                    break
            
            if latents_key:
                latents = torch.from_numpy(data[latents_key]).float()
                # 确保形状为 [batch_size, channels, height, width]
                if latents.dim() == 3:  # [channels, height, width]
                    latents = latents.unsqueeze(0)  # [1, channels, height, width]
                logger.debug(f"Loaded image features from {npz_path} (key: {latents_key}): {latents.shape}")
                self.feature_cache[npz_path] = latents
                return latents
            else:
                logger.warning(f"No latents found in {npz_path}, available keys: {list(data.keys())}")
                # 返回零特征作为fallback
                fallback_features = torch.zeros((1, self.image_feature_dim, 32, 32))
                self.feature_cache[npz_path] = fallback_features
                return fallback_features
                
        except Exception as e:
            logger.error(f"Failed to load image features from {npz_path}: {e}")
            # 返回零特征作为fallback
            fallback_features = torch.zeros((1, self.image_feature_dim, 32, 32))
            self.feature_cache[npz_path] = fallback_features
            return fallback_features
    
    def extract_sample_id_from_path(self, image_path: str) -> str:
        """
        从图像文件路径中提取样本ID
        
        Args:
            image_path: 图像文件路径，如 "./train/data/train/aircraft/44_0194x0282_sd3.npz"
            
        Returns:
            样本ID，如 "44"
        """
        filename = os.path.basename(image_path)
        # 去掉扩展名
        name_without_ext = os.path.splitext(filename)[0]
        # 如果文件名包含下划线和分辨率信息，提取第一个数字部分
        if '_' in name_without_ext:
            sample_id = name_without_ext.split('_')[0]
        else:
            sample_id = name_without_ext
        return sample_id


class PointCloudFeatureLoader:
    """3D点云特征加载器"""
    
    def __init__(self, pointcloud_dir: str, feature_dim: int = 64):
        """
        初始化点云特征加载器
        
        Args:
            pointcloud_dir: 点云特征文件目录
            feature_dim: 点云特征维度
        """
        self.pointcloud_dir = pointcloud_dir
        self.feature_dim = feature_dim
        self.feature_cache = {}
        
    def load_pointcloud_features(self, sample_id: str) -> torch.Tensor:
        """
        加载指定样本的3D点云特征
        
        Args:
            sample_id: 样本ID (如 "1", "2", "3")
            
        Returns:
            点云特征张量 [num_points, feature_dim]
        """
        if sample_id in self.feature_cache:
            return self.feature_cache[sample_id]
            
        # 尝试加载.pt文件
        pt_path = os.path.join(self.pointcloud_dir, f"{sample_id}.pt")
        if os.path.exists(pt_path):
            try:
                features = torch.load(pt_path, map_location='cpu')
                if isinstance(features, dict):
                    # 如果是字典，尝试提取特征
                    if 'features' in features:
                        features = features['features']
                    elif 'pointcloud' in features:
                        features = features['pointcloud']
                    else:
                        # 取第一个值
                        features = list(features.values())[0]
                
                # 确保是2D张量
                if features.dim() == 1:
                    features = features.unsqueeze(0)
                elif features.dim() > 2:
                    features = features.view(-1, features.shape[-1])
                
                logger.debug(f"Loaded pointcloud features for {sample_id}: {features.shape}")
                self.feature_cache[sample_id] = features
                return features
                
            except Exception as e:
                logger.error(f"Failed to load .pt file for {sample_id}: {e}")
        
        # 尝试加载.npy文件
        npy_path = os.path.join(self.pointcloud_dir, f"{sample_id}.npy")
        if os.path.exists(npy_path):
            try:
                features = torch.from_numpy(np.load(npy_path)).float()
                if features.dim() == 1:
                    features = features.unsqueeze(0)
                elif features.dim() > 2:
                    features = features.view(-1, features.shape[-1])
                
                logger.debug(f"Loaded pointcloud features for {sample_id}: {features.shape}")
                self.feature_cache[sample_id] = features
                return features
                
            except Exception as e:
                logger.error(f"Failed to load .npy file for {sample_id}: {e}")
        
        # 如果都失败了，返回零特征
        logger.warning(f"No pointcloud features found for {sample_id}, using zero features")
        fallback_features = torch.zeros((1, self.feature_dim))
        self.feature_cache[sample_id] = fallback_features
        return fallback_features

    def load_pointcloud_features_by_path(self, abs_path: str) -> torch.Tensor:
        """直接从绝对路径加载点云特征，支持 .pt/.npy。"""
        try:
            if abs_path.endswith('.pt') and os.path.exists(abs_path):
                features = torch.load(abs_path, map_location='cpu')
                if isinstance(features, dict):
                    if 'features' in features:
                        features = features['features']
                    elif 'pointcloud' in features:
                        features = features['pointcloud']
                    else:
                        features = list(features.values())[0]
                if features.dim() == 1:
                    features = features.unsqueeze(0)
                elif features.dim() > 2:
                    features = features.view(-1, features.shape[-1])
                return features
            if abs_path.endswith('.npy') and os.path.exists(abs_path):
                features = torch.from_numpy(np.load(abs_path)).float()
                if features.dim() == 1:
                    features = features.unsqueeze(0)
                elif features.dim() > 2:
                    features = features.view(-1, features.shape[-1])
                return features
        except Exception as e:
            logger.warning(f"Failed to load pointcloud by path {abs_path}: {e}")
        # 回退为零特征
        return torch.zeros((1, self.feature_dim))


class PointCloudFeatureProcessor(torch.nn.Module):
    """3D点云特征处理器"""
    
    def __init__(self, 
                 input_dim: int = 64,
                 output_dim: int = 768,  # 匹配CLIP-L的维度
                 max_points: int = 1000,
                 use_attention: bool = True):
        """
        初始化点云特征处理器
        
        Args:
            input_dim: 输入特征维度
            output_dim: 输出特征维度
            max_points: 最大点数
            use_attention: 是否使用注意力机制
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_points = max_points
        self.use_attention = use_attention
        
        # 特征投影层
        self.feature_projection = torch.nn.Linear(input_dim, output_dim)
        
        # 位置编码
        self.position_encoding = torch.nn.Parameter(
            torch.randn(max_points, output_dim) * 0.02
        )
        
        if use_attention:
            # 自注意力层
            self.attention = torch.nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=8,
                batch_first=True
            )
            self.norm = torch.nn.LayerNorm(output_dim)
        
        # 全局池化
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        
    def forward(self, pointcloud_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理点云特征
        
        Args:
            pointcloud_features: 点云特征 [batch_size, num_points, input_dim]
            
        Returns:
            (sequence_features, pooled_features): 序列特征和池化特征
        """
        if pointcloud_features.dim() == 2:
            # 如果是2维张量 [batch_size, input_dim]，添加点数维度
            batch_size, input_dim = pointcloud_features.shape
            pointcloud_features = pointcloud_features.unsqueeze(1)  # [batch_size, 1, input_dim]
            num_points = 1
        else:
            batch_size, num_points, _ = pointcloud_features.shape
        
        # 特征投影
        features = self.feature_projection(pointcloud_features)  # [B, N, D]
        
        # 添加位置编码
        if num_points <= self.max_points:
            pos_enc = self.position_encoding[:num_points].unsqueeze(0).expand(batch_size, -1, -1)
            features = features + pos_enc
        
        if self.use_attention:
            # 自注意力
            attn_output, _ = self.attention(features, features, features)
            features = self.norm(features + attn_output)
        
        # 全局池化
        pooled = self.global_pool(features.transpose(1, 2)).squeeze(-1)  # [B, D]
        
        return features, pooled


class ImageFeatureProcessor(torch.nn.Module):
    """图像特征处理器 - 处理图像潜在特征"""
    
    def __init__(self, 
                 input_channels: int = 32,
                 output_dim: int = 768,  # 匹配CLIP-L的维度
                 spatial_size: int = 32):
        """
        初始化图像特征处理器
        
        Args:
            input_channels: 输入通道数
            output_dim: 输出特征维度
            spatial_size: 空间尺寸
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.spatial_size = spatial_size
        
        # 卷积层提取空间特征
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((8, 8))  # 降采样到8x8
        )
        
        # 全局特征提取
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        
        # 特征投影到目标维度
        self.feature_projection = torch.nn.Linear(128, output_dim)
        
        # 空间特征投影
        self.spatial_projection = torch.nn.Linear(8 * 8 * 128, output_dim)  # 8x8=64个空间位置，每个128维
        
    def forward(self, image_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理图像特征
        
        Args:
            image_features: 图像特征 [batch_size, channels, height, width]
            
        Returns:
            (sequence_features, pooled_features): 序列特征和池化特征
        """
        batch_size = image_features.shape[0]
        
        # 卷积特征提取
        conv_features = self.conv_layers(image_features)  # [B, 128, 8, 8]
        
        # 全局池化特征
        global_pooled = self.global_pool(conv_features)  # [B, 128, 1, 1]
        global_features = global_pooled.flatten(1)  # [B, 128]
        pooled_features = self.feature_projection(global_features)  # [B, output_dim]
        
        # 空间序列特征
        spatial_features = conv_features.flatten(2).transpose(1, 2)  # [B, 64, 128]
        sequence_features = self.spatial_projection(spatial_features.flatten(1)).unsqueeze(1)  # [B, 1, output_dim]
        
        return sequence_features, pooled_features


class ThreeModalFeatureFusion(torch.nn.Module):
    """三模态特征融合器 - 融合文本、点云和图像特征"""
    
    def __init__(self, 
                 text_dim: int = 2048,  # CLIP-L + CLIP-G
                 pointcloud_dim: int = 768,
                 image_dim: int = 768,
                 fusion_dim: int = 2048,
                 fusion_method: str = "concat"):
        """
        初始化三模态特征融合器
        
        Args:
            text_dim: 文本特征维度
            pointcloud_dim: 点云特征维度
            image_dim: 图像特征维度
            fusion_dim: 融合后特征维度
            fusion_method: 融合方法 ("concat", "add", "attention")
        """
        super().__init__()
        self.text_dim = text_dim
        self.pointcloud_dim = pointcloud_dim
        self.image_dim = image_dim
        self.fusion_dim = fusion_dim
        self.fusion_method = fusion_method
        
        if fusion_method == "concat":
            # 三模态拼接融合
            total_dim = text_dim + pointcloud_dim + image_dim
            self.fusion_layer = torch.nn.Linear(total_dim, fusion_dim)
        elif fusion_method == "add":
            # 要求所有模态维度相同
            assert text_dim == pointcloud_dim == image_dim, "For add fusion, all dimensions must match"
            self.fusion_layer = torch.nn.Linear(text_dim, fusion_dim)
        elif fusion_method == "attention":
            # 改进的三模态注意力融合
            self.text_proj = torch.nn.Sequential(
                torch.nn.Linear(text_dim, fusion_dim),
                torch.nn.LayerNorm(fusion_dim),
                torch.nn.GELU()
            )
            self.pointcloud_proj = torch.nn.Sequential(
                torch.nn.Linear(pointcloud_dim, fusion_dim),
                torch.nn.LayerNorm(fusion_dim),
                torch.nn.GELU()
            )
            self.image_proj = torch.nn.Sequential(
                torch.nn.Linear(image_dim, fusion_dim),
                torch.nn.LayerNorm(fusion_dim),
                torch.nn.GELU()
            )
            
            # 多头注意力机制
            self.attention = torch.nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=8,  # 2048能被8整除 (2048/8=256)
                dropout=0.1,
                batch_first=True
            )
            
            # 位置编码（区分不同模态）
            self.position_embedding = torch.nn.Parameter(torch.randn(3, fusion_dim) * 0.02)
            
            # 层归一化
            self.norm1 = torch.nn.LayerNorm(fusion_dim)
            self.norm2 = torch.nn.LayerNorm(fusion_dim)
            
            # 前馈网络
            self.ffn = torch.nn.Sequential(
                torch.nn.Linear(fusion_dim, fusion_dim * 4),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(fusion_dim * 4, fusion_dim)
            )
            
            # 模态重要性权重生成器
            self.modality_weights = torch.nn.Sequential(
                torch.nn.Linear(fusion_dim * 3, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 3),
                torch.nn.Softmax(dim=-1)
            )
            
            # 残差连接的文本特征投影
            self.text_residual = torch.nn.Linear(text_dim, fusion_dim)
        elif fusion_method == "gentle":
            # 温和融合：主要保持文本特征，轻微融入3D信息
            self.pointcloud_proj_gentle = torch.nn.Sequential(
                torch.nn.Linear(pointcloud_dim, 512),
                torch.nn.LayerNorm(512),
                torch.nn.GELU(),
                torch.nn.Linear(512, text_dim)
            )
            self.image_proj_gentle = torch.nn.Sequential(
                torch.nn.Linear(image_dim, 256),
                torch.nn.LayerNorm(256),
                torch.nn.GELU(),
                torch.nn.Linear(256, text_dim)
            )
        elif fusion_method == "gated":
            # 门控融合：自适应学习何时使用3D信息
            # 特征投影网络
            self.pointcloud_proj_gated = torch.nn.Sequential(
                torch.nn.Linear(pointcloud_dim, 512),
                torch.nn.LayerNorm(512),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(512, 1024),
                torch.nn.LayerNorm(1024),
                torch.nn.GELU(),
                torch.nn.Linear(1024, text_dim)
            )
            self.image_proj_gated = torch.nn.Sequential(
                torch.nn.Linear(image_dim, 256),
                torch.nn.LayerNorm(256),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(256, 512),
                torch.nn.LayerNorm(512),
                torch.nn.GELU(),
                torch.nn.Linear(512, text_dim)
            )
            
            # 门控网络：学习如何融合特征
            self.gate_network = torch.nn.Sequential(
                torch.nn.Linear(text_dim * 3, 1024),  # 文本+点云+图像
                torch.nn.LayerNorm(1024),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(1024, 512),
                torch.nn.LayerNorm(512),
                torch.nn.GELU(),
                torch.nn.Linear(512, 3),  # 3个门控值：文本、点云、图像
                torch.nn.Softmax(dim=-1)
            )
            
            # 残差连接权重
            self.residual_weight = torch.nn.Parameter(torch.tensor(0.1))
        elif fusion_method == "attention_gated":
            # 基于注意力机制的门控融合
            # 特征投影网络（更深的网络）
            self.pointcloud_proj_attn = torch.nn.Sequential(
                torch.nn.Linear(pointcloud_dim, 512),
                torch.nn.LayerNorm(512),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(512, 1024),
                torch.nn.LayerNorm(1024),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(1024, text_dim)
            )
            self.image_proj_attn = torch.nn.Sequential(
                torch.nn.Linear(image_dim, 256),
                torch.nn.LayerNorm(256),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(256, 512),
                torch.nn.LayerNorm(512),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(512, text_dim)
            )
            
            # 多头自注意力机制：学习模态内部的重要性
            self.self_attention = torch.nn.MultiheadAttention(
                embed_dim=text_dim,
                num_heads=8,  # 8个注意力头
                dropout=0.1,
                batch_first=True
            )
            
            # 交叉注意力机制：学习模态间的交互
            self.cross_attention = torch.nn.MultiheadAttention(
                embed_dim=text_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            
            # 位置编码：区分不同模态
            self.modality_embedding = torch.nn.Parameter(torch.randn(3, text_dim) * 0.02)
            
            # 注意力门控网络：基于注意力输出计算门控权重
            self.attention_gate_network = torch.nn.Sequential(
                torch.nn.Linear(text_dim * 3, 1024),
                torch.nn.LayerNorm(1024),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(1024, 512),
                torch.nn.LayerNorm(512),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(512, 3),
                torch.nn.Softmax(dim=-1)
            )
            
            # 前馈网络：增强特征表达能力
            self.ffn = torch.nn.Sequential(
                torch.nn.Linear(text_dim, text_dim * 4),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(text_dim * 4, text_dim)
            )
            
            # 层归一化
            self.norm1 = torch.nn.LayerNorm(text_dim)
            self.norm2 = torch.nn.LayerNorm(text_dim)
            self.norm3 = torch.nn.LayerNorm(text_dim)
            
            # 残差连接权重
            self.residual_weight_attn = torch.nn.Parameter(torch.tensor(0.1))
        elif fusion_method == "image_guided":
            # 图像引导的保真融合：靠近原图语义，同时保持文本语义
            self.text_dim = text_dim
            self.pointcloud_dim = pointcloud_dim
            self.image_dim = image_dim

            # 投影到统一维度
            self.pc_proj_ig = torch.nn.Linear(pointcloud_dim, text_dim)
            self.img_proj_ig = torch.nn.Linear(image_dim, text_dim)

            # 归一化
            self.ln_text = torch.nn.LayerNorm(text_dim)
            self.ln_pc = torch.nn.LayerNorm(text_dim)
            self.ln_img = torch.nn.LayerNorm(text_dim)

            # 简单门控网络：产生 a(pc), b(img) ∈ [0,1]
            self.ig_gate = torch.nn.Sequential(
                torch.nn.Linear(text_dim * 3, 128),
                torch.nn.GELU(),
                torch.nn.Linear(128, 2),
                torch.nn.Sigmoid(),
            )
            # 目标相似度与最大推近系数（可通过环境变量调参）
            try:
                self.img_guide_cos_target = float(os.environ.get("IMG_GUIDE_COS_TARGET", "0.85"))
            except Exception:
                self.img_guide_cos_target = 0.85
            try:
                self.img_guide_max_gamma = float(os.environ.get("IMG_GUIDE_MAX_GAMMA", "0.5"))
            except Exception:
                self.img_guide_max_gamma = 0.5
        elif fusion_method == "hybrid_film_attn":
            # 混合式FiLM+注意力融合：多模态门控 + FiLM调制 + 图像引导
            self.text_dim = text_dim
            self.pointcloud_dim = pointcloud_dim
            self.image_dim = image_dim

            # 投影到统一维度
            self.pc_proj_hf = torch.nn.Linear(pointcloud_dim, text_dim)
            self.img_proj_hf = torch.nn.Linear(image_dim, text_dim)

            # 归一化
            self.ln_t = torch.nn.LayerNorm(text_dim)
            self.ln_p = torch.nn.LayerNorm(text_dim)
            self.ln_i = torch.nn.LayerNorm(text_dim)

            # 多模态门控（softmax得到 t/pc/img 三者权重）
            self.gate_hf = torch.nn.Sequential(
                torch.nn.Linear(text_dim * 3, 256),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(256, 3),
                torch.nn.Softmax(dim=-1),
            )

            # FiLM调制：输出 per-channel gamma/beta
            self.film_hf = torch.nn.Sequential(
                torch.nn.Linear(text_dim * 2, 512),  # [fused_pre, img]
                torch.nn.GELU(),
                torch.nn.Linear(512, text_dim * 2),  # [gamma, beta]
            )

            # 注意力细化（1层多头自注意力）
            self.attn_hf = torch.nn.MultiheadAttention(embed_dim=text_dim, num_heads=8, dropout=0.1, batch_first=True)
            self.norm_hf1 = torch.nn.LayerNorm(text_dim)
            self.ffn_hf = torch.nn.Sequential(
                torch.nn.Linear(text_dim, text_dim * 4),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(text_dim * 4, text_dim),
            )
            self.norm_hf2 = torch.nn.LayerNorm(text_dim)

            # 环境可调参数
            try:
                self.hybrid_gamma_scale = float(os.environ.get("HYBRID_GAMMA_SCALE", "0.5"))  # gamma幅度
            except Exception:
                self.hybrid_gamma_scale = 0.5
            try:
                self.hybrid_img_push_target = float(os.environ.get("HYBRID_IMG_PUSH_TARGET", "0.90"))
            except Exception:
                self.hybrid_img_push_target = 0.90
            try:
                self.hybrid_img_push_max = float(os.environ.get("HYBRID_IMG_PUSH_MAX", "0.5"))
            except Exception:
                self.hybrid_img_push_max = 0.5
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def forward(self, 
                text_features: torch.Tensor, 
                pointcloud_features: torch.Tensor,
                image_features: torch.Tensor) -> torch.Tensor:
        """
        融合文本、点云和图像特征
        
        Args:
            text_features: 文本特征 [batch_size, text_dim]
            pointcloud_features: 点云特征 [batch_size, pointcloud_dim]
            image_features: 图像特征 [batch_size, image_dim]
            
        Returns:
            融合后的特征 [batch_size, fusion_dim]
        """
        if self.fusion_method == "concat":
            # 改进的拼接融合：先统一维度再融合
            # 将点云和图像特征投影到文本特征维度
            if not hasattr(self, 'pointcloud_proj'):
                self.pointcloud_proj = torch.nn.Linear(self.pointcloud_dim, self.text_dim)
            if not hasattr(self, 'image_proj'):
                self.image_proj = torch.nn.Linear(self.image_dim, self.text_dim)
            
            # 投影到统一维度
            pointcloud_proj = self.pointcloud_proj(pointcloud_features)
            image_proj = self.image_proj(image_features)
            
            # 自适应权重融合（让模型学习平衡）
            if not hasattr(self, 'weight_gate'):
                self.weight_gate = torch.nn.Sequential(
                    torch.nn.Linear(self.text_dim * 3, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 3),
                    torch.nn.Softmax(dim=-1)
                )
            
            # 计算自适应权重
            combined_input = torch.cat([text_features, pointcloud_proj, image_proj], dim=-1)
            weights = self.weight_gate(combined_input)  # [B, 3]
            
            # 加权融合
            fused = (weights[:, 0:1] * text_features + 
                    weights[:, 1:2] * pointcloud_proj + 
                    weights[:, 2:3] * image_proj)
            
            return fused
        
        elif self.fusion_method == "add":
            # 三模态相加融合
            fused = text_features + pointcloud_features + image_features
            return self.fusion_layer(fused)
        
        elif self.fusion_method == "attention":
            # 改进的三模态注意力融合
            batch_size = text_features.size(0)
            
            # 1. 特征投影到统一维度
            text_proj = self.text_proj(text_features)  # [B, D]
            pointcloud_proj = self.pointcloud_proj(pointcloud_features)  # [B, D]
            image_proj = self.image_proj(image_features)  # [B, D]
            
            # 2. 添加位置编码
            text_with_pos = text_proj + self.position_embedding[0:1]  # [B, D]
            pointcloud_with_pos = pointcloud_proj + self.position_embedding[1:2]  # [B, D]
            image_with_pos = image_proj + self.position_embedding[2:3]  # [B, D]
            
            # 3. 堆叠成序列 [B, 3, D]
            features_seq = torch.stack([text_with_pos, pointcloud_with_pos, image_with_pos], dim=1)
            
            # 4. 多头自注意力
            attn_output, attn_weights = self.attention(features_seq, features_seq, features_seq)
            
            # 5. 残差连接和层归一化
            attn_output = self.norm1(attn_output + features_seq)
            
            # 6. 前馈网络
            ffn_output = self.ffn(attn_output)
            ffn_output = self.norm2(ffn_output + attn_output)
            
            # 7. 模态重要性权重计算
            # 使用原始投影特征计算权重（不包含位置编码）
            weight_input = torch.cat([text_proj, pointcloud_proj, image_proj], dim=-1)  # [B, D*3]
            modality_weights = self.modality_weights(weight_input)  # [B, 3]
            
            # 8. 加权融合
            weighted_features = (modality_weights[:, 0:1] * ffn_output[:, 0, :] +  # 文本
                               modality_weights[:, 1:2] * ffn_output[:, 1, :] +  # 点云
                               modality_weights[:, 2:3] * ffn_output[:, 2, :])   # 图像
            
            # 9. 残差连接（保持文本特征的主导地位）
            text_residual = self.text_residual(text_features)
            final_output = 0.7 * weighted_features + 0.3 * text_residual
            
            return final_output
        
        elif self.fusion_method == "gentle":
            # 温和融合：主要保持文本特征，轻微融入3D信息
            # 投影到文本特征维度
            pointcloud_proj = self.pointcloud_proj_gentle(pointcloud_features)
            image_proj = self.image_proj_gentle(image_features)
            
            # 可调注入系数（默认 0.02），支持环境变量 GENTLE_ALPHA（限制 [0,0.2]）
            try:
                alpha = float(os.environ.get("GENTLE_ALPHA", "0.02"))
            except Exception:
                alpha = 0.02
            alpha = max(0.0, min(0.2, alpha))
            beta_pc = alpha * 0.75
            beta_img = alpha * 0.25
            base = 1.0 - alpha

            fused = base * text_features + beta_pc * pointcloud_proj + beta_img * image_proj
            
            return fused
            
        elif self.fusion_method == "gated":
            # 门控融合：自适应学习何时使用3D信息
            # 1. 特征投影
            pointcloud_proj = self.pointcloud_proj_gated(pointcloud_features)
            image_proj = self.image_proj_gated(image_features)
            
            # 2. 计算门控权重
            # 将三个特征拼接作为门控网络的输入
            gate_input = torch.cat([text_features, pointcloud_proj, image_proj], dim=-1)
            gate_weights = self.gate_network(gate_input)  # [B, 3]
            
            # 3. 加权融合
            # gate_weights[:, 0] 对应文本权重
            # gate_weights[:, 1] 对应点云权重  
            # gate_weights[:, 2] 对应图像权重
            fused = (gate_weights[:, 0:1] * text_features + 
                    gate_weights[:, 1:2] * pointcloud_proj + 
                    gate_weights[:, 2:3] * image_proj)
            
            # 4. 残差连接（保持文本特征的主导地位）
            residual_output = fused + self.residual_weight * text_features
            
            return residual_output
            
        elif self.fusion_method == "attention_gated":
            # 基于注意力机制的门控融合：最先进的融合策略
            batch_size = text_features.size(0)
            
            # 1. 特征投影
            pointcloud_proj = self.pointcloud_proj_attn(pointcloud_features)
            image_proj = self.image_proj_attn(image_features)
            
            # 2. 添加模态位置编码
            text_with_pos = text_features + self.modality_embedding[0:1]  # [B, D]
            pointcloud_with_pos = pointcloud_proj + self.modality_embedding[1:2]  # [B, D]
            image_with_pos = image_proj + self.modality_embedding[2:3]  # [B, D]
            
            # 3. 堆叠成序列 [B, 3, D]
            features_seq = torch.stack([text_with_pos, pointcloud_with_pos, image_with_pos], dim=1)
            
            # 4. 自注意力：学习模态内部的重要性
            attn_output, attn_weights = self.self_attention(features_seq, features_seq, features_seq)
            attn_output = self.norm1(attn_output + features_seq)  # 残差连接
            
            # 5. 交叉注意力：学习模态间的交互
            # 以文本为query，视觉特征为key和value
            visual_features = torch.stack([pointcloud_with_pos, image_with_pos], dim=1)  # [B, 2, D]
            text_query = text_with_pos.unsqueeze(1)  # [B, 1, D]
            
            cross_attn_output, cross_attn_weights = self.cross_attention(
                text_query, visual_features, visual_features
            )
            cross_attn_output = self.norm2(cross_attn_output + text_query)  # 残差连接
            
            # 6. 前馈网络增强特征
            ffn_output = self.ffn(attn_output)
            ffn_output = self.norm3(ffn_output + attn_output)  # 残差连接
            
            # 7. 注意力门控网络：基于注意力输出计算门控权重
            # 使用原始特征（不包含位置编码）计算门控权重
            gate_input = torch.cat([text_features, pointcloud_proj, image_proj], dim=-1)
            gate_weights = self.attention_gate_network(gate_input)  # [B, 3]
            
            # 8. 加权融合：结合注意力输出和门控权重
            # 使用注意力增强后的特征进行融合
            fused = (gate_weights[:, 0:1] * ffn_output[:, 0, :] +  # 文本
                    gate_weights[:, 1:2] * ffn_output[:, 1, :] +  # 点云
                    gate_weights[:, 2:3] * ffn_output[:, 2, :])   # 图像
            
            # 9. 残差连接（保持文本特征的主导地位）
            residual_output = fused + self.residual_weight_attn * text_features
            
            return residual_output
        elif self.fusion_method == "image_guided":
            # 图像引导的保真融合
            pc_proj = self.pc_proj_ig(pointcloud_features)
            img_proj = self.img_proj_ig(image_features)

            # 归一化到统一分布
            t_ln = self.ln_text(text_features)
            p_ln = self.ln_pc(pc_proj)
            i_ln = self.ln_img(img_proj)

            # 门控权重 a, b ∈ [0,1]
            gate_in = torch.cat([t_ln, p_ln, i_ln], dim=-1)
            ab = self.ig_gate(gate_in)
            a = ab[:, 0:1]
            b = ab[:, 1:2]

            # 初步融合
            fused_pre = t_ln + a * p_ln + b * i_ln

            # 向图像方向推近（限制最大系数）
            def _cos(x, y, eps=1e-6):
                x_n = torch.nn.functional.normalize(x, dim=-1, eps=eps)
                y_n = torch.nn.functional.normalize(y, dim=-1, eps=eps)
                return (x_n * y_n).sum(dim=-1, keepdim=True)

            cos_ti = _cos(t_ln, i_ln)
            num = (self.img_guide_cos_target - cos_ti).clamp(min=0.0)
            den = (1.0 - cos_ti).clamp(min=1e-4)
            gamma = (num / den).clamp(min=0.0, max=self.img_guide_max_gamma)

            fused_dir = torch.nn.functional.normalize(fused_pre, dim=-1)
            img_dir = torch.nn.functional.normalize(i_ln, dim=-1)
            fused_refined_dir = torch.nn.functional.normalize((1.0 - gamma) * fused_dir + gamma * img_dir, dim=-1)
            fused = fused_refined_dir * fused_pre.norm(dim=-1, keepdim=True).clamp(min=1e-6)

            return fused
        elif self.fusion_method == "hybrid_film_attn":
            # 1) 统一维度并归一化
            p = self.pc_proj_hf(pointcloud_features)
            i = self.img_proj_hf(image_features)
            t = self.ln_t(text_features)
            p = self.ln_p(p)
            i = self.ln_i(i)

            # 2) 三模态门控融合（softmax权重）
            gates = self.gate_hf(torch.cat([t, p, i], dim=-1))  # [B,3]
            fused_pre = gates[:, 0:1] * t + gates[:, 1:2] * p + gates[:, 2:3] * i

            # 3) FiLM调制：基于[fused_pre, i]生成逐通道(gamma, beta)
            film_in = torch.cat([fused_pre, i], dim=-1)
            film_out = self.film_hf(film_in)
            gamma, beta = torch.chunk(film_out, 2, dim=-1)
            # 将gamma缩放到 [1 - s, 1 + s]
            gamma = 1.0 + self.hybrid_gamma_scale * torch.tanh(gamma)
            fused_film = gamma * fused_pre + beta

            # 4) 注意力细化（单token自注意力退化为恒等，但保留以兼容扩展）
            # 我们通过构造长度为1的序列做轻量化残差FFN
            x = fused_film.unsqueeze(1)  # [B,1,D]
            attn_out, _ = self.attn_hf(x, x, x)
            x = self.norm_hf1(x + attn_out)
            ffn_out = self.ffn_hf(x)
            x = self.norm_hf2(x + ffn_out)
            fused_refined = x.squeeze(1)

            # 5) 图像语义推近（余弦目标）
            def _cos(xy, yz, eps=1e-6):
                xn = torch.nn.functional.normalize(xy, dim=-1, eps=eps)
                yn = torch.nn.functional.normalize(yz, dim=-1, eps=eps)
                return (xn * yn).sum(dim=-1, keepdim=True)

            cos_fi = _cos(fused_refined, i)
            need = (self.hybrid_img_push_target - cos_fi).clamp(min=0.0)
            denom = (1.0 - cos_fi).clamp(min=1e-4)
            tau = (need / denom).clamp(max=self.hybrid_img_push_max)  # [0, max]

            fused_dir = torch.nn.functional.normalize(fused_refined, dim=-1)
            img_dir = torch.nn.functional.normalize(i, dim=-1)
            fused_guided = torch.nn.functional.normalize((1.0 - tau) * fused_dir + tau * img_dir, dim=-1)
            fused = fused_guided * fused_refined.norm(dim=-1, keepdim=True).clamp(min=1e-6)

            if os.environ.get("FUSION_DEBUG", "0") == "1":
                g = gates.mean(dim=0).tolist()
                logger.info(
                    f"FUSION_DEBUG: hybrid gates(t,pc,img)=[{g[0]:.2f},{g[1]:.2f},{g[2]:.2f}], "
                    f"avg_cos_fi={cos_fi.mean().item():.2f}, avg_tau={tau.mean().item():.2f}"
                )

            return fused
        elif self.fusion_method == "hierarchical":
            # 层次化融合：先融合点云和图像，再与文本融合
            if not hasattr(self, 'visual_fusion'):
                self.visual_fusion = torch.nn.Sequential(
                    torch.nn.Linear(self.pointcloud_dim + self.image_dim, 256),
                    torch.nn.LayerNorm(256),
                    torch.nn.GELU(),
                    torch.nn.Linear(256, 512)
                )
            
            if not hasattr(self, 'text_visual_fusion'):
                self.text_visual_fusion = torch.nn.Sequential(
                    torch.nn.Linear(self.text_dim + 512, 1024),
                    torch.nn.LayerNorm(1024),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(1024, self.fusion_dim)
                )
            
            if not hasattr(self, 'gate'):
                self.gate = torch.nn.Sequential(
                    torch.nn.Linear(self.text_dim + 512, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 1),
                    torch.nn.Sigmoid()
                )
            
            # 第一层：融合点云和图像特征
            visual_input = torch.cat([pointcloud_features, image_features], dim=-1)
            visual_fused = self.visual_fusion(visual_input)  # [B, 512]
            
            # 第二层：融合文本和视觉特征
            text_visual_input = torch.cat([text_features, visual_fused], dim=-1)
            
            # 门控机制：决定视觉信息的贡献程度
            gate_weight = self.gate(text_visual_input)  # [B, 1]
            
            # 最终融合
            fused_features = self.text_visual_fusion(text_visual_input)
            
            # 应用门控（保持文本特征的主导地位）
            final_output = gate_weight * fused_features + (1 - gate_weight) * text_features
            
            return final_output


class ThreeModalFusionStrategy:
    """三模态融合策略 - 整合所有组件"""
    
    def __init__(self, 
                 pointcloud_dir: str,
                 fusion_method: str = "concat",
                 pointcloud_dim: int = 64,
                 image_feature_dim: int = 32,
                 text_dim: int = 2048,
                 fusion_dim: int = 2048):
        """
        初始化三模态融合策略
        
        Args:
            pointcloud_dir: 3D点云特征目录
            fusion_method: 融合方法
            pointcloud_dim: 点云特征维度
            image_feature_dim: 图像特征维度
            text_dim: 文本特征维度
            fusion_dim: 融合后特征维度
        """
        self.pointcloud_dir = pointcloud_dir
        self.fusion_method = fusion_method
        
        # 初始化各个组件
        self.image_loader = ImageFeatureLoader(16)  # 实际图像特征通道数
        self.pointcloud_loader = PointCloudFeatureLoader(pointcloud_dir, pointcloud_dim)
        self.pointcloud_processor = PointCloudFeatureProcessor(
            input_dim=pointcloud_dim,
            output_dim=768
        )
        self.image_processor = ImageFeatureProcessor(
            input_channels=image_feature_dim,  # 使用传入的图像特征维度
            output_dim=768
        )
        self.fusion = ThreeModalFeatureFusion(
            text_dim=text_dim,
            pointcloud_dim=768,
            image_dim=768,
            fusion_dim=fusion_dim,
            fusion_method=fusion_method
        )
        # 全局注入强度（0.0~1.0），用于快速关闭/降低3D注入影响；默认1.0
        try:
            self.injection_scale = float(os.environ.get('FUSION_INJECTION_SCALE', '1.0'))
        except Exception:
            self.injection_scale = 1.0
        
        logger.info(f"Three-modal fusion strategy initialized with method: {fusion_method}")

        # 可选加载已保存的融合权重（用于推理复现训练时门控/投影参数）
        weights_path = os.environ.get('FUSION_WEIGHTS_LOAD', '')
        if weights_path and os.path.isfile(weights_path):
            try:
                self.load_weights(weights_path)
                logger.info(f"Loaded fusion weights from {weights_path}")
            except Exception as e:
                logger.warning(f"Failed to load fusion weights {weights_path}: {e}")
    
    def process_batch(self, 
                     text_features: torch.Tensor, 
                     image_paths: List[str]) -> torch.Tensor:
        """
        处理一个批次的样本，进行三模态特征融合
        
        Args:
            text_features: 文本特征 [batch_size, text_dim]
            image_paths: 图像文件路径列表
            
        Returns:
            融合后的特征 [batch_size, fusion_dim]
        """
        batch_size = text_features.shape[0]
        device = text_features.device
        
        # 确保所有模块都在正确的设备上（只在第一次或设备改变时调用）
        if not hasattr(self, '_device') or self._device != device:
            self.pointcloud_processor = self.pointcloud_processor.to(device)
            self.image_processor = self.image_processor.to(device)
            self.fusion = self.fusion.to(device)
            self._device = device
        
        # 处理每个样本
        pointcloud_features_list = []
        image_features_list = []
        
        for i, image_path in enumerate(image_paths):
            # 从图像路径提取样本ID
            sample_id = self.image_loader.extract_sample_id_from_path(image_path)
            # 兼容从 image_path 中携带的 pc=绝对路径（与训练/推理一致）
            pc_override = None
            if '|pc=' in image_path:
                try:
                    pc_override = image_path.split('|pc=', 1)[1]
                except Exception:
                    pc_override = None
            
            # 加载点云特征
            if pc_override and os.path.exists(pc_override):
                pointcloud_raw = self.pointcloud_loader.load_pointcloud_features_by_path(pc_override)
            else:
                pointcloud_raw = self.pointcloud_loader.load_pointcloud_features(sample_id)
            pointcloud_raw = pointcloud_raw.to(device)
            
            # 处理点云特征
            _, pointcloud_pooled = self.pointcloud_processor.forward(pointcloud_raw.unsqueeze(0))
            pointcloud_features_list.append(pointcloud_pooled.squeeze(0))
            
            # 加载图像特征（现在image_path是NPZ文件路径）
            if image_path.endswith('.npz'):
                # 加载NPZ图像特征文件
                image_raw = self.image_loader.load_image_features(image_path)
                image_raw = image_raw.to(device)
                _, image_pooled = self.image_processor.forward(image_raw)
                image_features_list.append(image_pooled.squeeze(0))
            elif image_path.startswith('missing_npz://'):
                # 严格模式：缺失NPZ直接报错
                if os.environ.get('FUSION_REQUIRE_NPZ', '0') == '1':
                    sample_id_dbg = self.image_loader.extract_sample_id_from_path(image_path)
                    raise RuntimeError(f"FUSION_REQUIRE_NPZ=1: missing NPZ for sample_id {sample_id_dbg}")
                # 非严格：使用零特征占位，保证仍走三模态融合
                zero_features = torch.zeros(pointcloud_pooled.squeeze(0).shape, device=device)
                image_features_list.append(zero_features)
            else:
                # 如果不是NPZ文件，使用零特征作为fallback
                zero_features = torch.zeros(pointcloud_pooled.squeeze(0).shape, device=device)
                image_features_list.append(zero_features)
        
        # 堆叠成批次
        pointcloud_features = torch.stack(pointcloud_features_list)  # [B, 768]
        image_features = torch.stack(image_features_list)  # [B, 768]
        
        # 三模态融合
        fused_features = self.fusion.forward(text_features, pointcloud_features, image_features)
        # 统一注入系数：scale=0 等价仅文本；scale=1 为原始融合输出
        if self.injection_scale != 1.0:
            fused_features = (1.0 - self.injection_scale) * text_features + self.injection_scale * fused_features
        
        logger.debug(f"Three-modal fusion: text={text_features.shape}, "
                    f"pointcloud={pointcloud_features.shape}, "
                    f"image={image_features.shape} -> {fused_features.shape}")
        
        return fused_features

    # ========= 权重保存/加载 =========
    def get_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """导出融合相关子模块权重（用于训练->推理复现）。"""
        sd: Dict[str, Dict[str, torch.Tensor]] = {}
        sd['pointcloud_processor'] = self.pointcloud_processor.state_dict()
        sd['image_processor'] = self.image_processor.state_dict()
        sd['fusion'] = self.fusion.state_dict()
        return sd

    def load_state_from_dict(self, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """从字典加载融合相关子模块权重。"""
        if 'pointcloud_processor' in state:
            self.pointcloud_processor.load_state_dict(state['pointcloud_processor'], strict=False)
        if 'image_processor' in state:
            self.image_processor.load_state_dict(state['image_processor'], strict=False)
        if 'fusion' in state:
            self.fusion.load_state_dict(state['fusion'], strict=False)

    def save_weights(self, path: str) -> None:
        """保存融合权重到文件（优先safetensors）。"""
        state = self.get_state_dict()
        if _HAS_SAFETENSORS and path.endswith('.safetensors'):
            # 将嵌套state展开为单层键以便保存
            flat = {}
            for k, v in state.items():
                for sk, sv in v.items():
                    flat[f"{k}.{sk}"] = sv.detach().cpu()
            safetensors_save(flat, path)
        else:
            torch.save(state, path)

    def load_weights(self, path: str) -> None:
        """从文件加载融合权重（支持safetensors/pt）。"""
        if _HAS_SAFETENSORS and path.endswith('.safetensors'):
            flat = safetensors_load(path)
            # 还原嵌套结构
            nested: Dict[str, Dict[str, torch.Tensor]] = {'pointcloud_processor': {}, 'image_processor': {}, 'fusion': {}}
            for k, v in flat.items():
                root, *rest = k.split('.')
                if root in nested:
                    nested[root]['.'.join(rest)] = v
            self.load_state_from_dict(nested)
        else:
            state = torch.load(path, map_location='cpu')
            self.load_state_from_dict(state)


def create_three_modal_fusion_strategy(pointcloud_dir: str,
                                      fusion_method: str = "concat",
                                      pointcloud_dim: int = 64,
                                      image_feature_dim: int = 32) -> ThreeModalFusionStrategy:
    """
    创建三模态融合策略的工厂函数
    
    Args:
        pointcloud_dir: 3D点云特征目录
        fusion_method: 融合方法
        pointcloud_dim: 点云特征维度
        image_feature_dim: 图像特征维度
        
    Returns:
        三模态融合策略实例
    """
    return ThreeModalFusionStrategy(
        pointcloud_dir=pointcloud_dir,
        fusion_method=fusion_method,
        pointcloud_dim=pointcloud_dim,
        image_feature_dim=image_feature_dim
    )
