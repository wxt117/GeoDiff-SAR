$train_mode = "sd3_lora"

# 三模态特征融合设置
$enable_3d_fusion = 1                   # 启用三模态特征融合
$pointcloud_dir = "./3d_features"       # 3D点云特征目录
$fusion_method = "hybrid_film_attn"     # 融合方法: concat, add, attention, gentle，gated
$pointcloud_dim = 64                    # 3D点云特征维度 (实际维度)
$image_feature_dim = 16                  # 图像特征维度

# Train data path | 设置训练用模型、图片
$pretrained_model = "./Stable-diffusion/SD3/sd3.5_medium.safetensors" # base model path | 底模路径
$vae = ""
$is_v2_model = 0 # SD2.0 model | SD2.0模型 2.0模型下 clip_skip 默认无效
$v_parameterization = 1 # parameterization | 参数化 v2 非512基础分辨率版本必须使用。
$train_data_dir = "./train/data/train" # train dataset path | 训练数据集路径
$reg_data_dir = ""	# reg dataset path | 正则数据集化路径
$network_weights = "" # pretrained weights for LoRA network | 若需要从已有的 LoRA 模型上继续训练，请填写 LoRA 模型路径。
$network_multiplier = 1.0 # lora权重倍数，默认1.0
$training_comment = "this LoRA model created from bdsqlsz by bdsqlsz'script" 
$dataset_class = ""
#$dataset_config = "./toml/datasets_qinglong.toml" # dataset config | 数据集配置文件路径
$disable_mmap_load_safetensors = 0 #在wsl下加载模型速度增加

#stable_cascade 训练相关参数
$effnet_checkpoint_path = "./VAE/effnet_encoder.safetensors" #effnet，相当于轻量化的VAE
$stage_c_checkpoint_path = "./Stable-diffusion/train/stage_c_bf16.safetensors" #stage_c，相当于base_model
$text_model_checkpoint_path = "" #te文本编码器，第一次默认不设置则自动从HF下载
$save_text_model = 1 #0关闭1开启，第一次训练设置保存TE的位置，之后不需要使用，只需要通过前面的参数text_model_checkpoint_path读取机壳
$previewer_checkpoint_path = "./Stable-diffusion/train/previewer.safetensors" #预览模型，开启预览图的话需要使用。
$adaptive_loss_weight = 1 #0关闭1开启，使用adaptive_loss_weight，官方推荐。关闭则使用P2LOSSWIGHT

#SD3 训练相关参数
$clip_l = "./clip/clip_l.safetensors"
$clip_g = "./clip/clip_g.safetensors"
$t5xxl = "./clip/t5xxl_fp16.safetensors"
$t5xxl_device = "cuda" #默认cuda，显存不够可改为CPU，但是很慢
$t5xxl_dtype = "bf16" #目前支持fp32、fp16、bf16
$text_encoder_batch_size = 16 #文本编码器批处理大小，4-16
$num_last_block_to_freeze = 0 #冻结最后几个block，默认0
$apply_t5_attn_mask = 1 # 是否应用T5的注意力掩码，默认为0
$clip_l_dropout_rate = 0.0 # clip_l dropout rate | clip_l dropout 率
$clip_g_dropout_rate = 0.0 # clip_g dropout rate | clip_g dropout 率
$t5_dropout_rate = 0.0 # t5 dropout rate | t5 dropout 率
$pos_emb_random_crop_rate = 0.0 # pos emb random crop rate | pos emb 随机裁剪率
$enable_scaled_pos_embed = 0 # enable scaled pos embed | 启用缩放 pos embed
$use_t5xxl_cache_only = 1 # 只使用T5缓存，不使用clip_l和clip_g

$blockwise_fused_optimizers = 0 # 是否使用块级融合优化器，默认为0
$blocks_to_swap = 0 # 交换的块数，默认为6

#flux 相关参数
$ae = $vae
$timestep_sampling = "flux_shift" # 时间步采样方法，可选 sd3用"sigma"、普通DDPM用"uniform" 或 flux用"sigmoid" 或者 "shift". shift需要修改discarete_flow_shift的参数
$discrete_flow_shift = 3.185 # Euler 离散调度器的离散流位移，sd3默认为3.0
$sigmoid_scale = 1.0 # sigmoid 采样的缩放因子，默认为 1.0。较大的值会使采样更加均匀
$model_prediction_type = "raw" # 模型预测类型，可选 flux的"raw"、增加噪声输入"additive" 或 sd选"sigma_scaled"
$guidance_scale = 1.0 # guidance scale，就是CFG, 默认为 1.0
$cpu_offload_checkpointing = 0 # 是否使用CPU卸载checkpoint，finetune默认开启
$mem_eff_save = 1 # 是否使用内存高效保存，默认为1
$split_qkv = 1 # 是否分离QKV，默认为1   原：0
$train_t5xxl = 1 #训练T5 原：0
$split_mode = 0 # 是否分离模式，默认为0, 开启后只训练单块减少显存加快训练速度。

#差异炼丹法
$base_weights = "" #指定合并到底模basemodel中的模型路径，多个用空格隔开。默认为空，不使用。
$base_weights_multiplier = "1.0" #指定合并模型的权重，多个用空格隔开，默认为1.0。

# Train related params | 训练相关参数
$resolution = "256,256" # image resolution w,h. 图片分辨率，宽,高。支持非正方形，但必须是 64 倍数。 原：1024,1024
$batch_size = 12 # batch size 一次性训练图片批处理数量，根据显卡质量对应调高。多GPU时建议减小batch_size
$max_train_epoches = 100 # max train epoches | 最大训练 epoch
$save_every_n_epochs = 5 # save every n epochs | 每 N 个 epoch 保存一次

$gradient_checkpointing = 0 #梯度检查，开启后可节约显存，但是速度变慢 原：1
$gradient_accumulation_steps = 1 # 梯度累加数量，变相放大batchsize的倍数 1
$optimizer_accumulation_steps = 0

$network_dim = 32 # network dim | 常用 4~128，不是越大越好
$network_alpha = 16 # network alpha | 常用与 network_dim 相同的值或者采用较小的值，如 network_dim的一半 防止下溢。默认值为 1，使用较小的 alpha 需要提升学习率。

$train_unet_only = 1 # train U-Net only | 仅训练 U-Net，开启这个会牺牲效果大幅减少显存使用。6G显存可以开启 原：0
$train_text_encoder_only = 0 # train Text Encoder only | 仅训练 文本编码器

$seed = 1026 # reproducable seed | 设置跑测试用的种子，输入一个prompt和这个种子大概率得到训练图。可以用来试触发关键词

#LORA_PLUS
$enable_lora_plus = 0 # 原：0
$loraplus_lr_ratio = 16
$loraplus_unet_lr_ratio = 16
$loraplus_text_encoder_lr_ratio = 4

#dropout | 抛出(目前和lycoris不兼容，请使用lycoris自带dropout)
$network_dropout = 0.1 # dropout 是机器学习中防止神经网络过拟合的技术，建议0.1~0.3 原：0
$scale_weight_norms = 1.0 #配合 dropout 使用，最大范数约束，推荐1.0
$rank_dropout = 0 #lora模型独创，rank级别的dropout，推荐0.1~0.3，未测试过多
$module_dropout = 0 #lora模型独创，module级别的dropout(就是分层模块的)，推荐0.1~0.3，未测试过多
$caption_dropout_every_n_epochs = 0 #dropout caption
$caption_dropout_rate = 0.1 #0~1 原：0
$caption_tag_dropout_rate = 0 #0~1

#noise | 噪声
$noise_offset = 0 # help allow SD to gen better blacks and whites，(0-1) | 帮助SD更好分辨黑白，推荐概念0.06，画风0.1
$adaptive_noise_scale = 0 #自适应偏移调整，10%~100%的noiseoffset大小
$noise_offset_random_strength = 0 #噪声随机强度
$multires_noise_iterations = 0 #多分辨率噪声扩散次数，推荐6-10,0禁用。
$multires_noise_discount = 0 #多分辨率噪声缩放倍数，推荐0.1-0.3,上面关掉的话禁用。
$min_snr_gamma = 5 #最小信噪比伽马值，减少低step时loss值，让学习效果更好。推荐3-5，5对原模型几乎没有太多影响，3会改变最终结果。修改为0禁用。 原：0
$ip_noise_gamma = 0 #误差噪声添加，防止误差累计
$ip_noise_gamma_random_strength = 0 #误差噪声随机强度
$debiased_estimation_loss = 0 #信噪比噪声修正，minsnr高级版
$loss_type = "l2" #损失函数类型，`smooth_l1`、`huber`、`l2`(就是MSE)
$huber_schedule = "snr" #huber调度器，可选 `exponential`、`constant` 或 `snr`
$huber_c = 0.1 #huber损失函数的c参数
$immiscible_noise = 0 #是否开启混合噪声


# Learning rate | 学习率
$lr = "1e-5"#原1e-5
$unet_lr = "1e-3"
$text_encoder_lr = "2e-5"
$lr_scheduler = "cosine_with_min_lr"
# "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup" | PyTorch自带6种动态学习率函数
# constant，常量不变, constant_with_warmup 线性增加后保持常量不变, linear 线性增加线性减少, polynomial 线性增加后平滑衰减, cosine 余弦波曲线, cosine_with_restarts 余弦波硬重启，瞬间最大值。
# 新增cosine_with_min_lr(适合训练lora)、warmup_stable_decay(适合训练db)、inverse_sqrt
$lr_warmup_steps = 0 # warmup steps | 学习率预热步数，lr_scheduler 为 constant 或 adafactor 时该值需要设为0。仅在 lr_scheduler 为 constant_with_warmup 时需要填写这个值
$lr_decay_steps = 0.1 # decay steps | 学习率衰减步数，仅在 lr_scheduler 为warmup_stable_decay时 需要填写，一般是10%总步数
$lr_scheduler_num_cycles = 1 # restarts nums | 余弦退火重启次数，仅在 lr_scheduler 为 cosine_with_restarts 时需要填写这个值
$lr_scheduler_timescale = 0 #times scale |时间缩放，仅在 lr_scheduler 为 inverse_sqrt 时需要填写这个值，默认同lr_warmup_steps
$lr_scheduler_min_lr_ratio = 0.1 #min lr ratio |最小学习率比率，仅在 lr_scheduler 为 cosine_with_min_lr、、warmup_stable_decay 时需要填写这个值，默认0

#optimizer | 优化器
$optimizer_type = "Prodigy"
# 可选优化器"adaFactor","AdamW","AdamW8bit","Lion","SGDNesterov","SGDNesterov8bit","DAdaptation",  
# 新增优化器"Lion8bit"(速度更快，内存消耗更少)、"DAdaptAdaGrad"、"DAdaptAdan"(北大最新算法，效果待测)、"DAdaptSGD"
# 新增DAdaptAdam、DAdaptLion、DAdaptAdanIP，强烈推荐DAdaptAdam
# 新增优化器"Sophia"(2倍速1.7倍显存)、"Prodigy"天才优化器，可自适应Dylora
# PagedAdamW8bit、PagedLion8bit、Adan、Tiger
# AdamWScheduleFree、SGDScheduleFree
# StableAdamW、Ranger
# came
# ScalableShampoo
# SOAP
# AdEMAMix8bit, PagedAdEMAMix8bit
$d_coef = "0.5" #prodigy D上升速度
$d0 = "1e-4" #dadaptation以及prodigy初始学习率
$fused_backward_pass = 0 #训练大模型float32精度专用节约显存，必须优化器adafactor或者adamw，gradient_accumulation_steps必须为1或者不开。
$fused_optimizer_groups = 0

#gorkfast | 快速拟合
$gradfilter_ema_alpha = 0 #EMA的动量超参数 设置ema_alpha来激活gradfilter_ema，推荐0.98，为0则关闭
$gradfilter_ema_lamb = 2.0 #滤波器ema的放大因子超参数。

# 数据集处理 打标captain相关
$shuffle_caption = 1 # 随机打乱tokens
$keep_tokens = 1 # keep heading N tokens when shuffling caption tokens | 在随机打乱 tokens 时，保留前 N 个不变。
$prior_loss_weight = 1 #正则化权重,0-1
$weighted_captions = 0 #权重打标，默认识别标签权重，语法同webui基础用法。例如(abc), [abc],(abc:1.23),但是不能在括号内加逗号，否则无法识别。一个文件最多75个tokens。
$secondary_separator = ";;;" #次要分隔符。被该分隔符分隔的部分将被视为一个token，并被洗牌和丢弃。然后由 caption_separator 取代。例如，如果指定 aaa;;bbb;;cc，它将被 aaa,bbb,cc 取代或一起丢弃。
$keep_tokens_separator = "|||" #批量保留不变，间隔符号
$enable_wildcard = 0 #通配符随机抽卡，格式参考 {aaa|bbb|ccc}，多行字幕可随机使用
$caption_prefix = "" #打标前缀，可以加入质量词如果底模需要，例如masterpiece, best quality,
$caption_suffix = "" #打标后缀，可以加入相机镜头如果需要，例如full body等
$alpha_mask = 0 #是否使用透明蒙版检测

# Output settings | 输出设置
$output_name = "sd35m-test-lora-1e-4" # output model name | 模型保存名称
$save_model_as = "safetensors" # model save ext | 模型保存格式 ckpt, pt, safetensors
$mixed_precision = "bf16" # 默认fp16,no,bf16可选
$save_precision = "bf16" # 默认fp16,fp32,bf16可选
$full_fp16 = 0 #开启全fp16模式，自动混合精度变为fp16，更节约显存
$full_bf16 = 1 #选择全bf16训练，必须30系以上显卡。
$fp8_base = 0 #开启fp8模式，更节约显存，实验性功能
$fp8_base_unet = 0 #开启fp8纯uent模式，更节约显存，实验性功能

# Resume training state | 恢复训练设置
$save_state = 0 # save training state | 保存训练状态 名称类似于 <output_name>-??????-state ?????? 表示 epoch 数
$resume = "" # resume from state | 从某个状态文件夹中恢复训练 需配合上方参数同时使用 由于规范文件限制 epoch 数和全局步数不会保存 即使恢复时它们也从 1 开始 与 network_weights 的具体实现操作并不一致
$save_state_on_train_end = 0 #只在训练结束最后保存训练状态

#保存toml文件
$output_config = 0 #开启后直接输出一个toml配置文件，但是无法同时训练，需要关闭才能正常训练。
$config_file = "./toml/" + $output_name + ".toml" #输出文件保存目录和文件名称，默认用模型保存同名。

#输出采样图片
$enable_sample = 1 #1开启出图，0禁用
$sample_at_first = 0 #是否在训练开始时就出图
$sample_every_n_epochs = 1 #每n个epoch出一次图
$sample_prompts = "./toml/qinglong.txt" #prompt文件路径
$sample_sampler = "euler_a" #采样器 'ddim', 'pndm', 'heun', 'dpmsolver', 'dpmsolver++', 'dpmsingle', 'k_lms', 'k_euler', 'k_euler_a', 'k_dpm_2', 'k_dpm_2_a'

#wandb 日志同步
$wandb_api_key = "" # wandbAPI KEY，用于登录

# 其他设置
$enable_bucket = 1 #开启分桶 原1（动态显存？）
$min_bucket_reso = 128 # arb min resolution | arb 最小分辨率 原：256
$max_bucket_reso = 512 # arb max resolution | arb 最大分辨率 原：2048
$bucket_no_upscale = 0 #分桶不放大 1
$persistent_workers = 1 # makes workers persistent, further reduces/eliminates the lag in between epochs. however it may increase memory usage | 跑的更快，吃内存。大概能提速2倍
$vae_batch_size = 4 #vae批处理大小，2-4
$clip_skip = 2 # clip skip | 玄学 一般用 2
$cache_latents = 1 #缓存潜变量
$cache_latents_to_disk = 1 # 缓存图片存盘，下次训练不需要重新缓存，1开启0禁用
$torch_compile = 0 #使用torch编译功能，需要版本大于2.1
$dynamo_backend = "inductor" #"eager", "aot_eager", "inductor","aot_ts_nvfuser","nvprims_nvfuser","cudagraphs","aot_torchxla_trace_once"用于训练
$TORCHINDUCTOR_FX_GRAPH_CACHE = 1 #启用本地 FX 图缓存。
$TORCHINDUCTOR_CACHE_DIR = "./torch_compile_cache" #指定所有磁盘缓存的位置。

#lycoris组件
$enable_lycoris = 0 # 开启lycoris
$conv_dim = 0 #卷积 dim，推荐＜32
$conv_alpha = 0 #卷积 alpha，推荐1或者0.3
$algo = "lokr" # algo参数，指定训练lycoris模型种类，
#包括lora(就是locon)、
#loha
#IA3
#lokr
#dylora
#full(DreamBooth先训练然后导出lora)
#diag-oft
#它通过训练适用于各层输出的正交变换来保留超球面能量。
#根据原始论文，它的收敛速度比 LoRA 更快，但仍需进行实验。
#dim 与区块大小相对应：我们在这里固定了区块大小而不是区块数量，以使其与 LoRA 更具可比性。

$dropout = 0 #lycoris专用dropout
$preset = "attn-mlp" #预设训练模块配置
#full: default preset, train all the layers in the UNet and CLIP|默认设置，训练所有Unet和Clip层
#full-lin: full but skip convolutional layers|跳过卷积层
#attn-mlp: train all the transformer block.|kohya配置，训练所有transformer模块
#attn-only：only attention layer will be trained, lot of papers only do training on attn layer.|只有注意力层会被训练，很多论文只对注意力层进行训练。
#unet-transformer-only： as same as kohya_ss/sd_scripts with disabled TE, or, attn-mlp preset with train_unet_only enabled.|和attn-mlp类似，但是关闭te训练
#unet-convblock-only： only ResBlock, UpSample, DownSample will be trained.|只训练卷积模块，包括res、上下采样模块
#./toml/example_lycoris.toml: 也可以直接使用外置配置文件，制定各个层和模块使用不同算法训练，需要输入位置文件路径，参考样例已添加。

$factor = 8 #只适用于lokr的因子，-1~8，8为全维度
$decompose_both = 0 #适用于lokr的参数，对 LoKr 分解产生的两个矩阵执行 LoRA 分解（默认情况下只分解较大的矩阵）
$block_size = 4 #适用于dylora,分割块数单位，最小1也最慢。一般4、8、12、16这几个选
$use_tucker = 0 #适用于除 (IA)^3 和full
$use_scalar = 0 #根据不同算法，自动调整初始权重
$train_norm = 0 #归一化层
$dora_wd = 1 #Dora方法分解，低rank使用。适用于LoRA, LoHa, 和LoKr
$full_matrix = 0  #全矩阵分解
$bypass_mode = 0 #通道模式，专为 bnb 8 位/4 位线性层设计。(QLyCORIS)适用于LoRA, LoHa, 和LoKr
$rescaled = 1 #适用于设置缩放，效果等同于OFT
$constrain = 0 #设置值为FLOAT，效果等同于COFT

#dylora组件
$enable_dylora = 0 # 开启dylora，和lycoris冲突，只能开一个。
$unit = 4 #分割块数单位，最小1也最慢。一般4、8、12、16这几个选

#Lora_FA
$enable_lora_fa = 0 # 开启lora_fa，和lycoris、dylora冲突，只能开一个。

#oft
$enable_oft = 0 # 开启oft，和以上冲突，只能开一个。

# block weights | 分层训练
$enable_block_weights = 0 #开启分层训练，和lycoris冲突，只能开一个。
$down_lr_weight = "1,0.2,1,1,0.2,1,1,0.2,1,1,1,1" #12层，需要填写12个数字，0-1.也可以使用函数写法，支持sine, cosine, linear, reverse_linear, zeros，参考写法down_lr_weight=cosine+.25 
$mid_lr_weight = "1"  #1层，需要填写1个数字，其他同上。
$up_lr_weight = "1,1,1,1,1,1,1,1,1,1,1,1"   #12层，同上上。
$block_lr_zero_threshold = 0  #如果分层权重不超过这个值，那么直接不训练。默认0。

$enable_block_dim = 0 #开启dim分层训练
$block_dims = "128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128" #dim分层，25层
$block_alphas = "16,16,32,16,32,32,64,16,16,64,64,64,16,64,16,64,32,16,16,64,16,16,16,64,16"  #alpha分层，25层
$conv_block_dims = "32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32" #convdim分层，25层
$conv_block_alphas = "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1" #convalpha分层，25层

# block lr
$enable_block_lr = 0
$block_lr = "0,$lr,$lr,0,$lr,$lr,0,$lr,$lr,0,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,0"

# flux block dim
$enable_flux_block_dim = 0
$img_attn_dim = $network_dim
$txt_attn_dim = $network_dim
$img_mlp_dim = $network_dim
$txt_mlp_dim = $network_dim
$img_mod_dim = 0
$txt_mod_dim = 0
$single_dim = $network_dim
$single_mod_dim = 0
$in_dims = "$network_dim,0,0,0,$network_dim"

# flux block layer
$enable_flux_block = 0
$train_single_block_indices = "all"
$train_double_block_indices = "all"

#SDXL专用参数
#https://www.bilibili.com/video/BV1tk4y137fo/
$min_timestep = 0 #最小时序，默认值0
$max_timestep = 1000 #最大时序，默认值1000
$cache_text_encoder_outputs = 0 #开启缓存文本编码器，开启后减少显存使用。但是无法和shuffle共用
$cache_text_encoder_outputs_to_disk = 0 #开启缓存文本编码器，开启后减少显存使用。但是无法和shuffle共用
$no_half_vae = 0 #禁止半精度，防止黑图。无法和mixed_precision混合精度共用。
$bucket_reso_steps = 32 #SDXL分桶可以选择32或者64。32更精细分桶。默认为64

#db checkpoint train
$stop_text_encoder_training = 0
$no_token_padding = 0 #不进行分词器填充

#sdxl_db
$diffusers_xformers = 0
$train_text_encoder = 0
$learning_rate_te1 = "5e-8"
$learning_rate_te2 = "5e-8"
$learning_rate_te3 = "5e-8"

#sdxl_cn3l or controlnet
$controlnet_model_name_or_path = ""
$conditioning_data_dir = ""
$cond_emb_dim = 32
$masked_loss = 0 #开启蒙版loss，对条件图处理，R通道255视为掩码mask，0视为无掩码

#多卡设置
$multi_gpu = 1                        #multi gpu | 多显卡训练开关，0关1开， 该参数仅限在显卡数 >= 2 使用
$highvram = 1                            #高显存模式，开启后会尽量使用显存  原0
$deepspeed = 0                         #deepspeed | 使用deepspeed训练，0关1开， 该参数仅限在显卡数 >= 2 使用
$zero_stage = 2                        #zero stage | zero stage 0,1,2,3,阶段2用于训练 该参数仅限在显卡数 >= 2 使用
$offload_optimizer_device = ""      #offload optimizer device | 优化器放置设备，cpu或者nvme, 该参数仅限在显卡数 >= 2 使用
$fp16_master_weights_and_gradients = 0 #fp16 master weights and gradients | fp16主权重和梯度，0关1开， 该参数仅限在显卡数 >= 2 使用

$ddp_timeout = 120 #ddp timeout | ddp超时时间，单位秒， 该参数仅限在显卡数 >= 2 使用
$ddp_gradient_as_bucket_view = 1 #ddp gradient as bucket view | ddp梯度作为桶视图，0关1开， 该参数仅限在显卡数 >= 2 使用
$ddp_static_graph = 1 #ddp static graph | ddp静态图，0关1开， 该参数仅限在显卡数 >= 2 使用

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
# Set CUDA devices for multi-GPU training
$Env:CUDA_VISIBLE_DEVICES = "1,2"

# Activate python venv
Set-Location $PSScriptRoot
if ($env:OS -ilike "*windows*") {
  if (Test-Path "./venv/Scripts/activate") {
    Write-Output "Windows venv"
    ./venv/Scripts/activate
  }
  elseif (Test-Path "./.venv/Scripts/activate") {
    Write-Output "Windows .venv"
    ./.venv/Scripts/activate
  }
}
elseif (Test-Path "./venv/bin/activate") {
  Write-Output "Linux venv"
  ./venv/bin/Activate.ps1
}
elseif (Test-Path "./.venv/bin/activate") {
  Write-Output "Linux .venv"
  ./.venv/bin/activate.ps1
}

$Env:HF_HOME = "huggingface"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$Env:HF_ENDPOINT = "https://hf-mirror.com" # 原：非注释
$Env:FUSION_INJECTION_SCALE = "1.0"
$Env:HYIMGTGT = "0.92"; if ($Env:HYIMGTGT) { $Env:HYBRID_IMG_PUSH_TARGET = $Env:HYIMGTGT } else { $Env:HYBRID_IMG_PUSH_TARGET = "0.92" }
$Env:HYBRID_IMG_PUSH_MAX = "0.7"
$Env:HYBRID_GAMMA_SCALE = "0.5"
$Env:FUSION_REQUIRE_NPZ = "1"
$Env:FUSION_WEIGHTS_STRICT = "1"
$network_module = "networks.lora"
$ext_args = [System.Collections.ArrayList]::new()
$launch_args = [System.Collections.ArrayList]::new()
$laungh_script = "train_network"

if ($train_mode -ieq "stable_cascade_lora") {
  $laungh_script = "stable_cascade_train_c_network"
}
elseif ($train_mode -ieq "flux_lora") {
  $network_module = $network_module + "_flux"
  if ($split_mode -ne 0 -or $split_qkv -ne 0 -or $train_t5xxl -ne 0 -or $enable_flux_block -ne 0 -or $enable_flux_block_dim -ne 0) {
    [void]$ext_args.Add("--network_args")
    if ($split_qkv -ne 0 -and $enable_lycoris -ne 1) {
      [void]$ext_args.Add("split_qkv=True")
    }
    if ($train_t5xxl -ne 0) {
      $cache_text_encoder_outputs = 0
      $cache_text_encoder_outputs_to_disk = 0
      [void]$ext_args.Add("train_t5xxl=True")
    }
    if ($split_mode -ne 0 -and $enable_lycoris -ne 1) {
      [void]$ext_args.Add("train_blocks=single")
      [void]$ext_args.Add("--split_mode")
    }
    if ($enable_flux_block_dim -ne 0) {
      [void]$ext_args.Add("img_attn_dim=$img_attn_dim")
      [void]$ext_args.Add("txt_attn_dim=$txt_attn_dim")
      [void]$ext_args.Add("img_mlp_dim=$img_mlp_dim")
      [void]$ext_args.Add("txt_mlp_dim=$txt_mlp_dim")
      [void]$ext_args.Add("img_mod_dim0=$img_mod_dim")
      [void]$ext_args.Add("txt_mod_dim=$txt_mod_dim")
      [void]$ext_args.Add("single_dim=$single_dim")
      [void]$ext_args.Add("single_mod_dim=$single_mod_dim")
      [void]$ext_args.Add("in_dims=$in_dims")
    }
    if ($enable_flux_block -ne 0) {
      [void]$ext_args.Add("train_single_block_indices=$train_single_block_indices")
      [void]$ext_args.Add("train_double_block_indices=$train_double_block_indices")
    }
  }
}
elseif ($train_mode -ieq "sd3_lora") {
  $network_module = $network_module + "_sd3"
  if ($split_mode -ne 0 -or $split_qkv -ne 0 -or $train_t5xxl -ne 0) {
    [void]$ext_args.Add("--network_args")
    if ($split_qkv -ne 0 -and $enable_lycoris -ne 1) {
      [void]$ext_args.Add("split_qkv=True")
    }
    if ($train_t5xxl -ne 0) {
      $cache_text_encoder_outputs = 0
      $cache_text_encoder_outputs_to_disk = 0
      [void]$ext_args.Add("train_t5xxl=True")
    }
    # if ($split_mode -ne 0 -and $enable_lycoris -ne 1) {
    #   [void]$ext_args.Add("train_blocks=single")
    #   [void]$ext_args.Add("--split_mode")
    # }
    # if ($enable_sd3_block_dim -ne 0) {
    #   [void]$ext_args.Add("in_dims=$in_dims")
    # }
    # if ($enable_sd3_block -ne 0) {
    #   [void]$ext_args.Add("train_single_block_indices=$train_single_block_indices")
    # }
  }
}

if ($multi_gpu -eq 1) {
  $launch_args += "--multi_gpu"
  $launch_args += "--mixed_precision=$mixed_precision"
  $launch_args += "--rdzv_backend=c10d"
  $blockwise_fused_optimizers = 0
  if ($deepspeed -eq 1) {
    [void]$ext_args.Add("--deepspeed")
    if ($zero_stage -ne 0) {
      [void]$ext_args.Add("--zero_stage=$zero_stage")
    }
    if ($offload_optimizer_device) {
      [void]$ext_args.Add("--offload_optimizer_device=$offload_optimizer_device")
    }
    if ($fp16_master_weights_and_gradients -eq 1) {
      [void]$ext_args.Add("--fp16_master_weights_and_gradients")
    }
  }
  if ($ddp_timeout -ne 0) {
    [void]$ext_args.Add("--ddp_timeout=$ddp_timeout")
  }
  if ($ddp_gradient_as_bucket_view -ne 0) {
    [void]$ext_args.Add("--ddp_gradient_as_bucket_view")
  }
  if ($ddp_static_graph -ne 0) {
    [void]$ext_args.Add("--ddp_static_graph")
  }
  if ($highvram -ne 0) {
    [void]$ext_args.Add("--highvram")
  }
}

if (-not ($train_mode -ilike "*lora")) {
  $network_module = ""
  $network_alpha = ""
  $conv_dim = ""
  $conv_alpha = ""
  $network_weights = ""
  $network_multiplier = 1.0
  $enable_block_weights = 0
  $enable_block_dim = 0
  $enable_lycoris = 0
  $enable_dylora = 0
  $enable_lora_fa = 0
  $enable_oft = 0
  $unet_lr = ""
  $text_encoder_lr = ""
  $train_unet_only = 0
  $train_text_encoder_only = 0
  $training_comment = ""
  $prior_loss_weight = 1
  $network_dropout = "0"
  $enable_lora_plus = 0
  $fp8_base_unet = 0
  if ($train_mode -ine "sdxl_cn3l") {
    $network_dim = ""
  }
}

if ($train_mode -ilike "*db") {
  if ($train_mode -ieq "db") {
    $laungh_script = "train_db";
    if ($no_token_padding -ne 0) {
      [void]$ext_args.Add("--no_token_padding")
    }
    if ($stop_text_encoder_training) {
      if ($gradient_accumulation_steps) {
        $stop_text_encoder_training = $stop_text_encoder_training * $gradient_accumulation_steps
      }
      [void]$ext_args.Add("--stop_text_encoder_training=$stop_text_encoder_training")
    }
    if ($learning_rate_te) {
      [void]$ext_args.Add("--learning_rate_te=$learning_rate_te")
    }
  }
  else {
    if ($train_mode -ieq "stable_cascade_db") {
      $laungh_script = "stable_cascade_train_stage_c"
      $pretrained_model = ""
      $learning_rate_te2 = 0
      $min_snr_gamma = 0
      $ip_noise_gamma = 0
      $loss_type = "l2"
      $weighted_captions = 0
      $debiased_estimation_loss = 0
      $immiscible_noise = 0
    }
    else {
      $laungh_script = "train"
      if ($train_mode -ieq "sdxl_db") {
        if ($diffusers_xformers -ne 0) {
          [void]$ext_args.Add("--diffusers_xformers")
        }
        if ($train_text_encoder -ne 0) {
          [void]$ext_args.Add("--train_text_encoder")
          if ($learning_rate_te1 -ne 0) {
            [void]$ext_args.Add("--learning_rate_te1=$learning_rate_te1")
          }
          if ($learning_rate_te2 -ne 0) {
            [void]$ext_args.Add("--learning_rate_te2=$learning_rate_te2")
          }
        }
        if ($enable_block_lr -ne 0) {
          [void]$ext_args.Add("--block_lr=$block_lr")   
        }
      }
      elseif ($train_mode -ieq "flux_db") {
        $mem_eff_save = 1
        if ($blockwise_fused_optimizers -ne 0) {
          [void]$ext_args.Add("--blockwise_fused_optimizers")
        }
        if ($blocks_to_swap -ne 0) {
          [void]$ext_args.Add("--blocks_to_swap=$blocks_to_swap")
        }
        if ($cpu_offload_checkpointing -ne 0) {
          [void]$ext_args.Add("--cpu_offload_checkpointing")
        }
        if ($mem_eff_save -ne 0) {
          [void]$ext_args.Add("--mem_eff_save")
        }
      }
      elseif ($train_mode -ieq "sd3_db") {
        if ($blockwise_fused_optimizers -ne 0) {
          [void]$ext_args.Add("--blockwise_fused_optimizers")
        }
        if ($blocks_to_swap -ne 0) {
          [void]$ext_args.Add("--blocks_to_swap=$blocks_to_swap")
        }
        if ($use_t5xxl_cache_only) {
          [void]$ext_args.Add("--use_t5xxl_cache_only")
        }
        if ($train_text_encoder -ne 0) {
          [void]$ext_args.Add("--train_text_encoder")
          if ($learning_rate_te1 -ne 0) {
            [void]$ext_args.Add("--learning_rate_te1=$learning_rate_te1")
          }
          if ($learning_rate_te2 -ne 0) {
            [void]$ext_args.Add("--learning_rate_te2=$learning_rate_te2")
          }
          if ($learning_rate_te3 -ne 0) {
            [void]$ext_args.Add("--learning_rate_te3=$learning_rate_te3")
          }
        }
      }
    }
  }
}

if ($train_mode -ilike "*cn3l" -or $train_mode -ilike "*controlnet") {
  if ($train_mode -ilike "*controlnet") {
    $laungh_script = "train_controlnet"
    if ($controlnet_model_name_or_path) {
      [void]$ext_args.Add("--controlnet_model_name_or_path=$controlnet_model_name_or_path")
    }
  }
  else {
    $laungh_script = "train_control_net_lllite"
    if ($cond_emb_dim) { 
      [void]$ext_args.Add("--cond_emb_dim=$cond_emb_dim")
    }
  }
  if ($conditioning_data_dir) { 
    [void]$ext_args.Add("--conditioning_data_dir=$conditioning_data_dir")
  }
  if ($masked_loss) { 
    [void]$ext_args.Add("--masked_loss")
  }
}

if ($train_mode -ilike "sdxl*") {
  $laungh_script = "sdxl_" + $laungh_script
  if ($min_timestep -ne 0) {
    [void]$ext_args.Add("--min_timestep=$min_timestep")
  }
  if ($max_timestep -ne 1000) {
    [void]$ext_args.Add("--max_timestep=$max_timestep")
  }
  if ($cache_text_encoder_outputs -ne 0) { 
    [void]$ext_args.Add("--cache_text_encoder_outputs")
    if ($cache_text_encoder_outputs_to_disk -ne 0) { 
      [void]$ext_args.Add("--cache_text_encoder_outputs_to_disk")
    }
    $shuffle_caption = 0
    $loraplus_text_encoder_lr_ratio = 0
    $caption_dropout_rate = 0
    $caption_tag_dropout_rate = 0
  }
  if ($no_half_vae -ne 0) { 
    [void]$ext_args.Add("--no_half_vae")
    $mixed_precision = ""
    $full_fp16 = 0
    $full_bf16 = 0
    $fp8_base = 0
    $fp8_base_unet = 0
  }
  if ($bucket_reso_steps -ne 64) { 
    [void]$ext_args.Add("--bucket_reso_steps=$bucket_reso_steps")
  }
}

if ($train_mode -ilike "sd3*" -or $train_mode -ilike "flux*") {
  if ($clip_l) {
    [void]$ext_args.Add("--clip_l=$clip_l")
  }
  if ($t5xxl) {
    [void]$ext_args.Add("--t5xxl=$t5xxl")
  }
  if ($apply_t5_attn_mask) {
    [void]$ext_args.Add("--apply_t5_attn_mask")
  }
  if ($num_last_block_to_freeze -ne 0) {
    [void]$ext_args.Add("--num_last_block_to_freeze=$num_last_block_to_freeze")
  }
  if ($train_mode -ilike "flux*") {
    $laungh_script = "flux_" + $laungh_script
    $enable_dylora = 0
    $enable_lora_fa = 0
    $enable_oft = 0
    $enable_block_lr = 0
    $enable_block_weights = 0
    $enable_block_dim = 0
    if ($timestep_sampling) {
      [void]$ext_args.Add("--timestep_sampling=$timestep_sampling")
    }
    if ($sigmoid_scale) {
      [void]$ext_args.Add("--sigmoid_scale=$sigmoid_scale")
    }
    if ($discrete_flow_shift) {
      [void]$ext_args.Add("--discrete_flow_shift=$discrete_flow_shift")
    }
    if ($model_prediction_type) {
      [void]$ext_args.Add("--model_prediction_type=$model_prediction_type")
    }
    if ($guidance_scale) {
      [void]$ext_args.Add("--guidance_scale=$guidance_scale")
    }
    if ($guidance_rescale -ne 0) {
      [void]$ext_args.Add("--guidance_rescale")
    }
    if ($ae) {
      [void]$ext_args.Add("--ae=$ae")
      $vae = ""
    }
    if ($torch_compile) {
      # $Env:CC = "E:\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64\cl.exe"
      [void]$ext_args.Add("--torch_compile")
      if ($dynamo_backend) {
        [void]$ext_args.Add("--dynamo_backend=$dynamo_backend")
      }
      if ($TORCHINDUCTOR_FX_GRAPH_CACHE -ne 0) {
        $Env:TORCHINDUCTOR_FX_GRAPH_CACHE = "1"
        if ($TORCHINDUCTOR_CACHE_DIR) {
          $Env:TORCHINDUCTOR_CACHE_DIR = $TORCHINDUCTOR_CACHE_DIR
        }
      }
    }
  }
  else {
    $laungh_script = "sd3_" + $laungh_script
    if ($clip_g) {
      [void]$ext_args.Add("--clip_g=$clip_g")
    }
    if ($t5xxl_device) {
      [void]$ext_args.Add("--t5xxl_device=$t5xxl_device")
    }
    # 三模态特征融合参数
    if ($enable_3d_fusion) {
      [void]$ext_args.Add("--enable_3d_fusion")
      if ($pointcloud_dir) {
        [void]$ext_args.Add("--pointcloud_dir=$pointcloud_dir")
      }
      if ($fusion_method) {
        [void]$ext_args.Add("--fusion_method=$fusion_method")
      }
      if ($pointcloud_dim) {
        [void]$ext_args.Add("--pointcloud_dim=$pointcloud_dim")
      }
      if ($image_feature_dim) {
        [void]$ext_args.Add("--image_feature_dim=$image_feature_dim")
      }
      # 训练融合网络参数
      [void]$ext_args.Add("--train_fusion")
      if ($fusion_lr) { [void]$ext_args.Add("--fusion_lr=$fusion_lr") }
    }
    if ($t5xxl_dtype) {
      [void]$ext_args.Add("--t5xxl_dtype=$t5xxl_dtype")
    }
    if ($text_encoder_batch_size) {
      [void]$ext_args.Add("--text_encoder_batch_size=$text_encoder_batch_size")
    }
    if ($clip_l_dropout_rate -ne 0) {
      [void]$ext_args.Add("--clip_l_dropout_rate=$clip_l_dropout_rate")
    }
    if ($clip_g_dropout_rate -ne 0) {
      [void]$ext_args.Add("--clip_g_dropout_rate=$clip_g_dropout_rate")
    }
    if ($t5_dropout_rate -ne 0) {
      [void]$ext_args.Add("--t5_dropout_rate=$t5_dropout_rate")
    }
    if ($pos_emb_random_crop_rate -ne 0) {
      [void]$ext_args.Add("--pos_emb_random_crop_rate=$pos_emb_random_crop_rate")
    }
    if ($enable_scaled_pos_embed) {
      [void]$ext_args.Add("--enable_scaled_pos_embed")
    }
  }
  if ($cache_text_encoder_outputs -ne 0) { 
    [void]$ext_args.Add("--cache_text_encoder_outputs")
    if ($cache_text_encoder_outputs_to_disk -ne 0) { 
      [void]$ext_args.Add("--cache_text_encoder_outputs_to_disk")
    }
    $shuffle_caption = 0
    $loraplus_text_encoder_lr_ratio = 0
    $caption_dropout_rate = 0
    $caption_tag_dropout_rate = 0
  }
  [void]$ext_args.Add("--sdpa")
}
elseif ($torch_compile) {
  [void]$ext_args.Add("--sdpa")
  [void]$ext_args.Add("--torch_compile")
  if ($dynamo_backend) {
    [void]$ext_args.Add("--dynamo_backend=$dynamo_backend")
  }
  if ($TORCHINDUCTOR_FX_GRAPH_CACHE -ne 0) {
    $Env:TORCHINDUCTOR_FX_GRAPH_CACHE = "1"
    if ($TORCHINDUCTOR_CACHE_DIR) {
      $Env:TORCHINDUCTOR_CACHE_DIR = $TORCHINDUCTOR_CACHE_DIR
    }
  }
}
else {
  [void]$ext_args.Add("--xformers")
}

if ($train_mode -ilike "stable_cascade*") {
  if ($effnet_checkpoint_path) {
    [void]$ext_args.Add("--effnet_checkpoint_path=$effnet_checkpoint_path")
  }
  if ($stage_c_checkpoint_path) {
    [void]$ext_args.Add("--stage_c_checkpoint_path=$stage_c_checkpoint_path")
  }
  if ($text_model_checkpoint_path) {
    [void]$ext_args.Add("--text_model_checkpoint_path=$text_model_checkpoint_path")
  }
  if ($save_text_model -ne 0) {
    [void]$ext_args.Add("--save_text_model")
  }
  if ($previewer_checkpoint_path) {
    [void]$ext_args.Add("--previewer_checkpoint_path=$previewer_checkpoint_path")
  }
  if ($adaptive_loss_weight -ne 0) {
    [void]$ext_args.Add("--adaptive_loss_weight")
  }
}

if ($dataset_class) { 
  [void]$ext_args.Add("--dataset_class=$dataset_class")
}
elseif ($dataset_config) {
  [void]$ext_args.Add("--dataset_config=$dataset_config")
}
else {
  [void]$ext_args.Add("--train_data_dir=$train_data_dir")
  if ($reg_data_dir) {
    [void]$ext_args.Add("--reg_data_dir=$reg_data_dir")
  }
  if ($batch_size) {
    [void]$ext_args.Add("--train_batch_size=$batch_size")
  }
  if ($resolution) {
    [void]$ext_args.Add("--resolution=$resolution")
  }
  if ($enable_bucket) {
    [void]$ext_args.Add("--enable_bucket")
    [void]$ext_args.Add("--min_bucket_reso=$min_bucket_reso")
    [void]$ext_args.Add("--max_bucket_reso=$max_bucket_reso")
    if ($bucket_no_upscale) {
      [void]$ext_args.Add("--bucket_no_upscale")
    }
  }
}

if ($caption_tag_dropout_rate) {
  [void]$ext_args.Add("--caption_tag_dropout_rate=$caption_tag_dropout_rate")
}

if ($pretrained_model) {
  [void]$ext_args.Add("--pretrained_model_name_or_path=$pretrained_model")
}

if ($vae) {
  [void]$ext_args.Add("--vae=$vae")
}

if ($disable_mmap_load_safetensors) {
  [void]$ext_args.Add("--disable_mmap_load_safetensors")
}

if ($save_model_as) {
  [void]$ext_args.Add("--save_model_as=$save_model_as")
}

if ($is_v2_model) {
  [void]$ext_args.Add("--v2")
  $min_snr_gamma = 0
  $debiased_estimation_loss = 0
  if ($v_parameterization) {
    [void]$ext_args.Add("--v_parameterization")
    [void]$ext_args.Add("--scale_v_pred_loss_like_noise_pred")
    [void]$ext_args.Add("--zero_terminal_snr")
  }
}
elseif ($train_mode -ilike "hunyuan*") {
  $laungh_script = "hunyuan_" + $laungh_script
  $min_snr_gamma = 0
  $debiased_estimation_loss = 0
  [void]$ext_args.Add("--v_parameterization")
  [void]$ext_args.Add("--scale_v_pred_loss_like_noise_pred")
  [void]$ext_args.Add("--zero_terminal_snr")
}
else {
  [void]$ext_args.Add("--clip_skip=$clip_skip")
}

if ($prior_loss_weight -and $prior_loss_weight -ne 1) {
  [void]$ext_args.Add("--prior_loss_weight=$prior_loss_weight")
}

if ($network_dim) {
  [void]$ext_args.Add("--network_dim=$network_dim")
}

if ($network_alpha) {
  [void]$ext_args.Add("--network_alpha=$network_alpha")
}

if ($training_comment) {
  [void]$ext_args.Add("--training_comment=$training_comment")
}

if ($persistent_workers) {
  [void]$ext_args.Add("--persistent_data_loader_workers")
}

if ($max_data_loader_n_workers) {
  [void]$ext_args.Add("--max_data_loader_n_workers=$max_data_loader_n_workers")
}

if ($shuffle_caption) {
  [void]$ext_args.Add("--shuffle_caption")
}

if ($weighted_captions) {
  [void]$ext_args.Add("--weighted_captions")
}

if ($cache_latents) { 
  [void]$ext_args.Add("--cache_latents")
  if ($cache_latents_to_disk) {
    [void]$ext_args.Add("--cache_latents_to_disk")
  }
}

if ($output_config) {
  [void]$ext_args.Add("--output_config")
  [void]$ext_args.Add("--config_file=$config_file")
}

if ($gradient_checkpointing) {
  [void]$ext_args.Add("--gradient_checkpointing")
}

if ($save_state -eq 1) {
  [void]$ext_args.Add("--save_state")
  if ($save_state_on_train_end -eq 1) {
    [void]$ext_args.Add("--save_state_on_train_end")
  }
}

if ($resume) {
  [void]$ext_args.Add("--resume=$resume")
}

if ($noise_offset -ne 0) {
  [void]$ext_args.Add("--noise_offset=$noise_offset")
  if ($adaptive_noise_scale) {
    [void]$ext_args.Add("--adaptive_noise_scale=$adaptive_noise_scale")
  }
  if ($noise_offset_random_strength) {
    [void]$ext_args.Add("--noise_offset_random_strength")
  }
}
elseif ($multires_noise_iterations -ne 0) {
  [void]$ext_args.Add("--multires_noise_iterations=$multires_noise_iterations")
  [void]$ext_args.Add("--multires_noise_discount=$multires_noise_discount")
}

if ($network_dropout -ne 0) {
  $enable_lycoris = 0
  [void]$ext_args.Add("--network_dropout=$network_dropout")
  if ($scale_weight_norms -ne 0) { 
    [void]$ext_args.Add("--scale_weight_norms=$scale_weight_norms")
  }
  if ($enable_dylora -ne 0) {
    [void]$ext_args.Add("--network_args")
    if ($rank_dropout) {
      [void]$ext_args.Add("rank_dropout=$rank_dropout")
    }
    if ($module_dropout) {
      [void]$ext_args.Add("module_dropout=$module_dropout")
    }
  }
}

if ($enable_block_weights) {
  [void]$ext_args.Add("--network_args")
  [void]$ext_args.Add("down_lr_weight=$down_lr_weight")
  [void]$ext_args.Add("mid_lr_weight=$mid_lr_weight")
  [void]$ext_args.Add("up_lr_weight=$up_lr_weight")
  [void]$ext_args.Add("block_lr_zero_threshold=$block_lr_zero_threshold")
  if ($enable_block_dim) {
    [void]$ext_args.Add("block_dims=$block_dims")
    [void]$ext_args.Add("block_alphas=$block_alphas")
    if ($conv_block_dims) {
      [void]$ext_args.Add("conv_block_dims=$conv_block_dims")
      if ($conv_block_alphas) {
        [void]$ext_args.Add("conv_block_alphas=$conv_block_alphas")
      }
    }
    elseif ($conv_dim) {
      [void]$ext_args.Add("conv_dim=$conv_dim")
      if ($conv_alpha) {
        [void]$ext_args.Add("conv_alpha=$conv_alpha")
      }
    }
  }
}
elseif ($enable_lycoris) {
  $network_module = "lycoris.kohya"
  [void]$ext_args.Add("--network_args")
  [void]$ext_args.Add("algo=$algo")
  if ($algo -ine "ia3" -and $algo -ine "diag-oft") {
    if ($algo -ine "full") {
      if ($conv_dim) {
        [void]$ext_args.Add("conv_dim=$conv_dim")
        if ($conv_alpha) {
          [void]$ext_args.Add("conv_alpha=$conv_alpha")
        }
      }
      if ($use_tucker) {
        [void]$ext_args.Add("use_tucker=True")
      }
      if ($algo -ine "dylora") {
        if ($dora_wd) {
          [void]$ext_args.Add("dora_wd=True")
        }
        if ($bypass_mode) {
          [void]$ext_args.Add("bypass_mode=True")
        }
        if ($use_scalar) {
          [void]$ext_args.Add("use_scalar=True")
        }
      }
    }
    [void]$ext_args.Add("preset=$preset")
  }
  if ($dropout -and $algo -ieq "locon") {
    [void]$ext_args.Add("dropout=$dropout")
  }
  if ($train_norm -and $algo -ine "ia3") {
    [void]$ext_args.Add("train_norm=True")
  }
  if ($algo -ieq "lokr") {
    [void]$ext_args.Add("factor=$factor")
    if ($decompose_both) {
      [void]$ext_args.Add("decompose_both=True")
    }
    if ($full_matrix) {
      [void]$ext_args.Add("full_matrix=True")
    }
  }
  elseif ($algo -ieq "dylora") {
    [void]$ext_args.Add("block_size=$block_size")
  }
  elseif ($algo -ieq "diag-oft") {
    if ($rescaled) {
      [void]$ext_args.Add("rescaled=True")
    }
    if ($constrain) {
      [void]$ext_args.Add("constrain=$constrain")
    }
  }
}
elseif ($enable_dylora) {
  $network_module = "networks.dylora"
  [void]$ext_args.Add("--network_args")
  [void]$ext_args.Add("unit=$unit")
  if ($conv_dim) {
    [void]$ext_args.Add("conv_dim=$conv_dim")
    if ($conv_alpha) {
      [void]$ext_args.Add("conv_alpha=$conv_alpha")
    }
  }
  if ($module_dropout) {
    [void]$ext_args.Add("module_dropout=$module_dropout")
  }
}
elseif ($enable_lora_fa) {
  $network_module = "networks.lora_fa"
}
elseif ($enable_oft) {
  $network_module = "networks.oft"
}
else {
  if ($conv_dim) {
    [void]$ext_args.Add("--network_args")
    [void]$ext_args.Add("conv_dim=$conv_dim")
    if ($conv_alpha) {
      [void]$ext_args.Add("conv_alpha=$conv_alpha")
    }
  }
}

if ($optimizer_type -ieq "adafactor") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("scale_parameter=False")
  [void]$ext_args.Add("warmup_init=False")
  [void]$ext_args.Add("relative_step=False")
  if ($lr_scheduler -and $lr_scheduler -ine "constant") {
    $lr_warmup_steps = 100
  }
}

if ($optimizer_type -ilike "DAdapt*") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
  if ($optimizer_type -ieq "DAdaptation" -or $optimizer_type -ilike "DAdaptAdam*") {
    [void]$ext_args.Add("decouple=True")
    if ($optimizer_type -ieq "DAdaptAdam") {
      [void]$ext_args.Add("use_bias_correction=True")
    }
  }
  $lr = "1"
  if ($unet_lr) {
    $unet_lr = $lr
  }
  if ($text_encoder_lr) {
    $text_encoder_lr = $lr
  }
}

if ($optimizer_type -ieq "Lion" -or $optimizer_type -ieq "Lion8bit" -or $optimizer_type -ieq "PagedLion8bit") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
  [void]$ext_args.Add("betas=.95,.98")
}

if ($optimizer_type -ieq "AdamW8bit") {
  $optimizer_type = ""
  [void]$ext_args.Add("--use_8bit_adam")
}

if ($optimizer_type -ieq "PagedAdamW8bit" -or $optimizer_type -ieq "AdamW") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
  [void]$ext_args.Add("betas=.9,.95")
}

if ($optimizer_type -ieq "PagedAdEMAMix8bit" -or $optimizer_type -ieq "AdEMAMix8bit") {
  [void]$ext_args.Add("--optimizer_type=bitsandbytes.optim.$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ieq "Sophia") {
  [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.SophiaH")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ieq "Prodigy") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
  [void]$ext_args.Add("betas=.9,.99")
  [void]$ext_args.Add("decouple=True")
  [void]$ext_args.Add("use_bias_correction=True")
  [void]$ext_args.Add("d_coef=$d_coef")
  if ($lr_warmup_steps) {
    [void]$ext_args.Add("safeguard_warmup=True")
  }
  if ($d0) {
    [void]$ext_args.Add("d0=$d0")
  }
  $lr = "1"
  if ($unet_lr) {
    $unet_lr = $lr
  }
  if ($text_encoder_lr) {
    $text_encoder_lr = $lr
  }
}

if ($optimizer_type -ieq "Ranger") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  if (-not($train_unet_only -or $train_text_encoder_only) -or $train_text_encoder) {
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("decouple_lr=True")
  }
}

if ($optimizer_type -ieq "Adan") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  if (-not($train_unet_only -or $train_text_encoder_only) -or $train_text_encoder) {
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("decouple_lr=True")
  }
}

if ($optimizer_type -ieq "StableAdamW") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  if (-not($train_unet_only -or $train_text_encoder_only) -or $train_text_encoder) {
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("decouple_lr=True")
  }
}

if ($optimizer_type -ieq "Tiger") {
  [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.Tiger")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ilike "*ScheduleFree") {
  $lr_scheduler = ""
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.08")
  #[void]$ext_args.Add("weight_lr_power=0.001")
}

if ($optimizer_type -ieq "adammini") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ieq "adamg") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.1")
  [void]$ext_args.Add("weight_decouple=True")
}

if ($optimizer_type -ieq "came") {
  [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.CAME")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ieq "SOAP") {
  [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.SOAP")
}

if ($optimizer_type -ieq "sara") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
  [void]$ext_args.Add("threshold=2e-3")
}

if ($gradfilter_ema_alpha -ne 0) {
  [void]$ext_args.Add("--gradfilter_ema_alpha=$gradfilter_ema_alpha")
  [void]$ext_args.Add("--gradfilter_ema_lamb=$gradfilter_ema_lamb")
}

if ($unet_lr) {
  if ($train_unet_only) {
    $train_text_encoder_only = 0
    $loraplus_text_encoder_lr_ratio = 0
    [void]$ext_args.Add("--network_train_unet_only")
  }
  [void]$ext_args.Add("--unet_lr=$unet_lr")
}

if ($text_encoder_lr) {
  if ($train_text_encoder_only) {
    $loraplus_unet_lr_ratio = 0
    [void]$ext_args.Add("--network_train_text_encoder_only")
  }
  [void]$ext_args.Add("--text_encoder_lr=$text_encoder_lr")
}

if ($enable_lora_plus) {
  [void]$ext_args.Add("--network_args")
  if ($loraplus_unet_lr_ratio) {
    [void]$ext_args.Add("loraplus_lr_ratio=$loraplus_unet_lr_ratio")
  }
  elseif ($loraplus_text_encoder_lr_ratio -eq 0) {
    [void]$ext_args.Add("loraplus_lr_ratio=$loraplus_lr_ratio")
  }
  if ($loraplus_text_encoder_lr_ratio) {
    [void]$ext_args.Add("loraplus_text_encoder_lr_ratio=$loraplus_text_encoder_lr_ratio")
  }
}

if ($network_weights) {
  [void]$ext_args.Add("--network_weights=$network_weights")
}

if ($network_multiplier -ne 1.0) {
  [void]$ext_args.Add("--network_multiplier=$network_multiplier")
}

if ($keep_tokens) {
  [void]$ext_args.Add("--keep_tokens=$keep_tokens")
}

if ($keep_tokens_separator) {
  [void]$ext_args.Add("--keep_tokens_separator=$keep_tokens_separator")
}

if ($secondary_separator) {
  [void]$ext_args.Add("--secondary_separator=$secondary_separator")
}

if ($enable_wildcard) {
  [void]$ext_args.Add("--enable_wildcard")
}

if ($caption_prefix) {
  [void]$ext_args.Add("--caption_prefix=$caption_prefix")
}

if ($caption_suffix) {
  [void]$ext_args.Add("--caption_suffix=$caption_suffix")
}

if ($alpha_mask) {
  [void]$ext_args.Add("--alpha_mask")
}

if ($min_snr_gamma -ne 0) {
  [void]$ext_args.Add("--min_snr_gamma=$min_snr_gamma")
}
elseif ($debiased_estimation_loss -ne 0) {
  [void]$ext_args.Add("--debiased_estimation_loss")
}

if ($loss_type -ne "l2") {
  [void]$ext_args.Add("--loss_type=$loss_type")
  if ($huber_schedule -ne "snr") {
    [void]$ext_args.Add("--huber_schedule=$huber_schedule")
  }
  if ($huber_c -ne 0.1) {
    [void]$ext_args.Add("--huber_c=$huber_c")
  }
}

if ($immiscible_noise) {
  [void]$ext_args.Add("--immiscible_noise")
}

if ($ip_noise_gamma -ne 0) {
  [void]$ext_args.Add("--ip_noise_gamma=$ip_noise_gamma")
  if ($ip_noise_gamma_random_strength) {
    [void]$ext_args.Add("--ip_noise_gamma_random_strength")
  }
}

if ($wandb_api_key) {
  [void]$ext_args.Add("--wandb_api_key=$wandb_api_key")
  [void]$ext_args.Add("--log_with=wandb")
  [void]$ext_args.Add("--log_tracker_name=" + $output_name)
}

if ($enable_sample) {
  if ($sample_at_first) {
    [void]$ext_args.Add("--sample_at_first")
  }
  [void]$ext_args.Add("--sample_every_n_epochs=$sample_every_n_epochs")
  [void]$ext_args.Add("--sample_prompts=$sample_prompts")
  [void]$ext_args.Add("--sample_sampler=$sample_sampler")
}

if ($base_weights) {
  [void]$ext_args.Add("--base_weights")
  foreach ($base_weight in $base_weights.Split(" ")) {
    [void]$ext_args.Add($base_weight)
  }
  [void]$ext_args.Add("--base_weights_multiplier")
  foreach ($ratio in $base_weights_multiplier.Split(" ")) {
    [void]$ext_args.Add([float]$ratio)
  }
}

if ($fused_backward_pass -ne 0) {
  [void]$ext_args.Add("--fused_backward_pass")
  $gradient_accumulation_steps = 0
}
elseif ($fused_optimizer_groups) {
  [void]$ext_args.Add("--fused_optimizer_groups")
}

if ($fp8_base -ne 0) {
  [void]$ext_args.Add("--fp8_base")
}
if ($fp8_base_unet -ne 0) {
  [void]$ext_args.Add("--fp8_base_unet")
}
if ($full_fp16 -ne 0) {
  [void]$ext_args.Add("--full_fp16")
  $mixed_precision = "fp16"
  $save_precision = "fp16"
}
elseif ($full_bf16 -ne 0) {
  [void]$ext_args.Add("--full_bf16")
  $mixed_precision = "bf16"
  $save_precision = "bf16"
}

if ($mixed_precision) {
  [void]$ext_args.Add("--mixed_precision=$mixed_precision")
}

if ($network_module) {
  [void]$ext_args.Add("--network_module=$network_module")
}

if ($gradient_accumulation_steps) {
  [void]$ext_args.Add("--gradient_accumulation_steps=$gradient_accumulation_steps")
}

if ($optimizer_accumulation_steps) {
  [void]$ext_args.Add("--optimizer_accumulation_steps=$optimizer_accumulation_steps")
}

if ($lr_scheduler) {
  [void]$ext_args.Add("--lr_scheduler=$lr_scheduler")
}

if ($lr_scheduler_num_cycles) {
  [void]$ext_args.Add("--lr_scheduler_num_cycles=$lr_scheduler_num_cycles")
}

if ($lr_warmup_steps) {
  [void]$ext_args.Add("--lr_warmup_steps=$lr_warmup_steps")
}

if ($lr_decay_steps) {
  [void]$ext_args.Add("--lr_decay_steps=$lr_decay_steps")
}

if ($lr_scheduler_timescale) {
  [void]$ext_args.Add("--lr_scheduler_timescale=$lr_scheduler_timescale")
}

if ($lr_scheduler_min_lr_ratio) {
  [void]$ext_args.Add("--lr_scheduler_min_lr_ratio=$lr_scheduler_min_lr_ratio")
}

if ($caption_dropout_every_n_epochs) {
  [void]$ext_args.Add("--caption_dropout_every_n_epochs=$caption_dropout_every_n_epochs")
}
if ($caption_dropout_rate) {
  [void]$ext_args.Add("--caption_dropout_rate=$caption_dropout_rate")
}
if ($caption_tag_dropout_rate) {
  [void]$ext_args.Add("--caption_tag_dropout_rate=$caption_tag_dropout_rate")
}

# 在启动训练前，保存本次三模态融合配置，供推理复用
try {
  $gentle_alpha = 0.02
  if ($Env:GENTLE_ALPHA) {
    $gentle_alpha = [double]$Env:GENTLE_ALPHA
  }
  $fusion_config = @{
    enable_3d_fusion      = [bool]$enable_3d_fusion
    fusion_method         = $fusion_method
    pointcloud_dim        = [int]$pointcloud_dim
    image_feature_dim     = [int]$image_feature_dim
    gentle_alpha          = $gentle_alpha
    fusion_seed           = 1026
    use_t5xxl_cache_only  = [bool]$use_t5xxl_cache_only
  }
  $fusion_dump = "./output/" + $output_name + "_fusion.json"
  ($fusion_config | ConvertTo-Json -Depth 3) | Out-File -FilePath $fusion_dump -Encoding UTF8
  Write-Output ("Saved fusion config -> " + $fusion_dump)
} catch {
  Write-Warning ("Save fusion config failed: " + $_)
}

# run train
python -m accelerate.commands.launch --num_cpu_threads_per_process=8 --main_process_port=29501 $launch_args "./sd-scripts/sd3_train_network_with_3d.py" `
  --output_dir="./output" `
  --logging_dir="./logs" `
  --max_train_epochs=$max_train_epoches `
  --learning_rate=$lr `
  --output_name=$output_name `
  --save_every_n_epochs=$save_every_n_epochs `
  --save_precision=$save_precision `
  --seed=$seed  `
  --max_token_length=225 `
  --caption_extension=".txt" `
  --vae_batch_size=$vae_batch_size `
  $ext_args

Write-Output "Train finished"
Read-Host | Out-Null ;
