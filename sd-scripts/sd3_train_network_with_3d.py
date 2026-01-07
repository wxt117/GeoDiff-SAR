#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SD3训练网络 - 支持3D点云特征融合
基于原始sd3_train_network.py，添加3D点云特征融合功能
"""

import argparse
import copy
import math
import os
import random
from typing import Any, Optional

import torch
from accelerate import Accelerator
from library import sd3_models, strategy_sd3_with_3d as strategy_sd3, utils
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import flux_models, flux_train_utils, flux_utils, sd3_train_utils, sd3_utils, strategy_base, strategy_sd3_with_3d as strategy_sd3, train_util
import train_network
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class Sd3NetworkTrainerWith3D(train_network.NetworkTrainer):
    """SD3网络训练器 - 支持三模态特征融合"""

    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None


    def train(self, args):
        """训练主函数"""
        # 验证三模态融合参数
        if args.enable_3d_fusion:
            if not args.pointcloud_dir:
                raise ValueError("启用三模态融合时必须指定pointcloud_dir")
            if not os.path.exists(args.pointcloud_dir):
                raise ValueError(f"3D点云特征目录不存在: {args.pointcloud_dir}")
            logger.info(f"三模态特征融合已启用: {args.pointcloud_dir}, 方法: {args.fusion_method}")

        # 调用父类训练方法
        return super().train(args)

    def load_target_model(self, args, weight_dtype, accelerator):
        """加载目标模型"""
        # 目前卸载到CPU以节省内存
        loading_dtype = None if args.fp8_base else weight_dtype

        # 如果文件是fp8且我们使用fp8_base，我们可以直接加载它（fp8）
        state_dict = utils.load_safetensors(
            args.pretrained_model_name_or_path, "cpu", disable_mmap=args.disable_mmap_load_safetensors, dtype=loading_dtype
        )
        mmdit = sd3_utils.load_mmdit(state_dict, loading_dtype, "cpu")
        self.model_type = mmdit.model_type
        mmdit.set_pos_emb_random_crop_rate(args.pos_emb_random_crop_rate)

        # 为位置嵌入设置分辨率
        if args.enable_scaled_pos_embed:
            latent_sizes = [round(math.sqrt(res[0] * res[1])) // 8 for res in self.resolutions]  # 8是潜在空间的步长
            latent_sizes = list(set(latent_sizes))  # 移除重复项
            logger.info(f"为分辨率准备缩放位置嵌入: {self.resolutions}, 大小: {latent_sizes}")
            mmdit.enable_scaled_pos_embed(True, latent_sizes)

        if args.fp8_base:
            # 检查模型数据类型
            if mmdit.dtype == torch.float8_e4m3fnuz or mmdit.dtype == torch.float8_e5m2 or mmdit.dtype == torch.float8_e5m2fnuz:
                raise ValueError(f"不支持的fp8模型数据类型: {mmdit.dtype}")
            elif mmdit.dtype == torch.float8_e4m3fn:
                logger.info("加载了fp8 SD3模型")

        clip_l = sd3_utils.load_clip_l(
            args.clip_l, weight_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors, state_dict=state_dict
        )
        clip_l.eval()
        clip_g = sd3_utils.load_clip_g(
            args.clip_g, weight_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors, state_dict=state_dict
        )
        clip_g.eval()

        # 如果文件是fp8且我们使用fp8_base（不是unet），我们可以直接加载它（fp8）
        if args.fp8_base and not args.fp8_base_unet:
            loading_dtype = None  # 保持原样
        else:
            loading_dtype = weight_dtype

        # 将t5xxl加载到CPU需要很长时间，将来应该加载到GPU
        t5xxl = sd3_utils.load_t5xxl(
            args.t5xxl, loading_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors, state_dict=state_dict
        )
        t5xxl.eval()
        if args.fp8_base and not args.fp8_base_unet:
            # 检查数据类型
            if t5xxl.dtype == torch.float8_e4m3fnuz or t5xxl.dtype == torch.float8_e5m2 or t5xxl.dtype == torch.float8_e5m2fnuz:
                raise ValueError(f"不支持的fp8模型数据类型: {t5xxl.dtype}")
            elif t5xxl.dtype == torch.float8_e4m3fn:
                logger.info("加载了fp8 T5XXL模型")

        vae = sd3_utils.load_vae(
            args.vae, weight_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors, state_dict=state_dict
        )

        return mmdit.model_type, [clip_l, clip_g, t5xxl], vae, mmdit

    def get_tokenize_strategy(self, args):
        """获取分词策略"""
        logger.info(f"t5xxl_max_token_length: {args.t5xxl_max_token_length}")
        from library import strategy_sd3 as original_strategy_sd3
        return original_strategy_sd3.Sd3TokenizeStrategy(args.t5xxl_max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy):
        """获取分词器"""
        return [tokenize_strategy.clip_l, tokenize_strategy.clip_g, tokenize_strategy.t5xxl]

    def get_latents_caching_strategy(self, args):
        """获取潜在空间缓存策略"""
        # 使用原始的strategy_sd3模块
        from library import strategy_sd3 as original_strategy_sd3
        latents_caching_strategy = original_strategy_sd3.Sd3LatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        return latents_caching_strategy

    def get_text_encoding_strategy(self, args):
        """获取文本编码策略 - 支持三模态融合"""
        return strategy_sd3.Sd3TextEncodingStrategyWith3D(
            args.apply_lg_attn_mask,
            args.apply_t5_attn_mask,
            args.clip_l_dropout_rate,
            args.clip_g_dropout_rate,
            args.t5_dropout_rate,
            pointcloud_dir=args.pointcloud_dir if args.enable_3d_fusion else None,
            enable_3d_fusion=args.enable_3d_fusion,
            fusion_method=args.fusion_method,
            pointcloud_dim=args.pointcloud_dim,
            image_feature_dim=args.image_feature_dim,
        )

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        """后处理网络"""
        # 检查t5xxl是否被训练
        self.train_t5xxl = network.train_t5xxl

        if self.train_t5xxl and args.cache_text_encoder_outputs:
            raise ValueError(
                "T5XXL正在训练，所以不能使用cache_text_encoder_outputs / T5XXL学習時はcache_text_encoder_outputsは使用できません"
            )

        # 获取权重数据类型
        weight_dtype, _ = train_util.prepare_dtype(args)
        
        # 将文本编码器移动到设备
        text_encoders[0].to(accelerator.device, dtype=weight_dtype)
        text_encoders[1].to(accelerator.device, dtype=weight_dtype)
        text_encoders[2].to(accelerator.device)

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        """获取噪声调度器"""
        # shift 3.0是默认值
        noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=3.0, sigma_max_scale=args.sigma_max_scale
        )
        self.noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        return noise_scheduler

    def encode_images_to_latents(self, args, accelerator, vae, images):
        """将图像编码为潜在表示"""
        return vae.encode(images)

    def shift_scale_latents(self, args, latents):
        """缩放潜在表示"""
        return sd3_models.SDVAE.process_in(latents)

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet: flux_models.Flux,
        network,
        weight_dtype,
        train_unet,
    ):
        """获取噪声预测和目标"""
        # 采样我们将添加到潜在表示的噪声
        noise = torch.randn_like(latents)

        # 获取噪声模型输入和时间步
        noisy_model_input, timesteps, sigmas = sd3_train_utils.get_noisy_model_input_and_timesteps(
            args, self.noise_scheduler_copy, latents, noise, accelerator.device, weight_dtype
        )

        # 确保隐藏状态需要梯度
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for t in text_encoder_conds:
                if t is not None and t.dtype.is_floating_point:
                    t.requires_grad_(True)

        # 预测噪声残差
        lg_out, t5_out, lg_pooled, l_attn_mask, g_attn_mask, t5_attn_mask = text_encoder_conds
        text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
        context, lg_pooled = text_encoding_strategy.concat_encodings(lg_out, t5_out, lg_pooled)
        if not args.apply_lg_attn_mask:
            l_attn_mask = None
            g_attn_mask = None
        if not args.apply_t5_attn_mask:
            t5_attn_mask = None

        # 调用模型
        with accelerator.autocast():
            # TODO 支持注意力掩码
            model_pred = unet(noisy_model_input, timesteps, context=context, y=lg_pooled)

        # 遵循: https://arxiv.org/abs/2206.00364 的第5节
        # 模型输出的预处理
        model_pred = model_pred * (-sigmas) + noisy_model_input

        # 这些权重方案使用统一的时间步采样
        # 而是对损失进行后加权
        weighting = sd3_train_utils.compute_loss_weighting_for_sd3(
            weighting_scheme=args.weighting_scheme, sigmas=sigmas
        )

        # 流匹配损失
        target = latents

        # 差分输出保持
        if "custom_attributes" in batch:
            diff_output_pr_indices = []
            for i, custom_attributes in enumerate(batch["custom_attributes"]):
                if "diff_output_preservation" in custom_attributes and custom_attributes["diff_output_preservation"]:
                    diff_output_pr_indices.append(i)

            if len(diff_output_pr_indices) > 0:
                network.set_multiplier(0.0)
                with torch.no_grad(), accelerator.autocast():
                    model_pred_prior = unet(
                        noisy_model_input[diff_output_pr_indices],
                        timesteps[diff_output_pr_indices],
                        context=context[diff_output_pr_indices],
                        y=lg_pooled[diff_output_pr_indices],
                    )
                network.set_multiplier(1.0)  # 可能在下一步被"network_multipliers"覆盖

                model_pred_prior = model_pred_prior * (-sigmas[diff_output_pr_indices]) + noisy_model_input[diff_output_pr_indices]

                # 差分输出保持不需要权重，因为已经应用了

                target[diff_output_pr_indices] = model_pred_prior.to(target.dtype)

        return model_pred, target, timesteps, None, weighting

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        """后处理损失"""
        return loss

    def get_sai_model_spec(self, args):
        """获取SAI模型规范"""
        return train_util.get_sai_model_spec(None, args, False, True, False, sd3=self.model_type)

    def update_metadata(self, metadata, args):
        """更新元数据"""
        metadata["ss_apply_lg_attn_mask"] = args.apply_lg_attn_mask
        metadata["ss_apply_t5_attn_mask"] = args.apply_t5_attn_mask
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        # 添加三模态融合相关元数据
        if args.enable_3d_fusion:
            metadata["ss_enable_3d_fusion"] = True
            metadata["ss_fusion_method"] = args.fusion_method
            metadata["ss_pointcloud_dim"] = args.pointcloud_dim
            metadata["ss_image_feature_dim"] = args.image_feature_dim

    def is_text_encoder_not_needed_for_training(self, args):
        """检查训练是否需要文本编码器"""
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        """准备文本编码器梯度检查点变通方法"""
        if index == 0 or index == 1:  # CLIP-L/CLIP-G
            return super().prepare_text_encoder_grad_ckpt_workaround(index, text_encoder)
        else:  # T5XXL
            return super().prepare_text_encoder_grad_ckpt_workaround(index, text_encoder)

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, mmdit):
        """采样图像"""
        text_encoders = text_encoder  # 为了兼容性
        text_encoders = self.get_models_for_text_encoding(args, accelerator, text_encoders)

        sd3_train_utils.sample_images(
            accelerator, args, epoch, global_step, mmdit, vae, text_encoders, self.sample_prompts_te_outputs
        )


def setup_parser() -> argparse.ArgumentParser:
    """设置参数解析器"""
    parser = train_network.setup_parser()
    sd3_train_utils.add_sd3_training_arguments(parser)
    
    # 添加三模态特征融合参数
    parser.add_argument("--enable_3d_fusion", action="store_true", help="启用三模态特征融合")
    parser.add_argument("--pointcloud_dir", type=str, default=None, help="3D点云特征目录")
    parser.add_argument("--fusion_method", type=str, default="concat", 
                      choices=["concat", "add", "attention", "gentle", "gated", "attention_gated", "image_guided", "film", "hybrid_film_attn"], help="三模态特征融合方法")
    parser.add_argument("--pointcloud_dim", type=int, default=64, help="3D点云特征维度")
    parser.add_argument("--image_feature_dim", type=int, default=32, help="图像特征维度")
    # 训练融合网络参数
    parser.add_argument("--train_fusion", action="store_true", help="同时训练融合网络参数")
    parser.add_argument("--fusion_lr", type=float, default=None, help="融合网络学习率（默认等于 --learning_rate）")
    
    return parser


def main():
    """主函数"""
    parser = setup_parser()
    
    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    
    trainer = Sd3NetworkTrainerWith3D()
    trainer.train(args)


if __name__ == "__main__":
    main()
