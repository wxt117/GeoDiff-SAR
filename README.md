# GeoDiff-SAR: A Geometric Prior Guided Diffusion Model for SAR Image Generation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-green.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-See%20LICENSE-orange.svg)](LICENSE)

This repository contains the official implementation of **GeoDiff-SAR**, a diffusion model framework for Synthetic Aperture Radar (SAR) image generation. The implementation is based on Stable Diffusion 3.5 (SD3.5) with LoRA fine-tuning, achieving high-quality SAR image generation through **tri-modal feature fusion** (text, 3D point cloud, and image features).

## 📋 Overview

GeoDiff-SAR is a novel approach to SAR image generation that leverages geometric priors from 3D models. This implementation **directly uses pre-extracted 3D model features** without requiring scattering calculations, simplifying the feature extraction pipeline. By fusing text descriptions, 3D geometric features, and image features, we achieve high-fidelity SAR image generation.

The overall architecture of the proposed GeoDiff-SAR framework is shown below. (a) Construction of Geometric Priors: 3D models are processed via a point cloud processing module to extract explicit geometric scattering characteristics, such as multi-bounce reflections, serving as a robust physical prior. (b) Multi-modal Fusion Network: These physical features are synthesized with textual and visual conditions through a multi-modal fusion network to condition the Stable Diffusion 3.5 backbone, which is efficiently adapted to the SAR domain using LoRA (trainable) while keeping the pre-trained weights
frozen. (c) Controllable Generative Model: As depicted in the top-right, the generated high-fidelity SAR images are utilized to augment scarce real training data for downstream PyTorch Image Models Multi-Label Classification tasks, thereby validating the practical utility and effectiveness of the proposed data augmentation strategy.

<img width="1422" height="617" alt="fig-overall" src="https://github.com/user-attachments/assets/b334bf0a-1928-4c93-9f77-18e7159986c3" />

### ✨ Key Features

- 🎯 **Tri-Modal Feature Fusion**: Seamlessly integrates text, 3D point cloud, and image features
- 🚀 **Direct 3D Feature Usage**: Pre-extracted 3D model features without scattering computation
- 🔧 **Multiple Fusion Strategies**: Supports 8 different feature fusion methods
- 💡 **Parameter-Efficient Fine-Tuning**: Lightweight LoRA-based training approach
- 🔄 **End-to-End Pipeline**: Complete training and inference code provided

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for acceleration)
- At least 24GB GPU memory (for training)

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd for_github
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

For CUDA support, install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 1: Download Model Weights

Download the following model weights and place them in the corresponding directories:

```
Stable-diffusion/SD3/
  └── sd3.5_medium.safetensors

clip/
  ├── clip_l.safetensors
  ├── clip_g.safetensors
  └── t5xxl_fp16.safetensors
```

**Download links**:
- SD3.5 Medium: [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3-medium)
- CLIP-L/G: Included in SD3.5 model files
- T5-XXL: [Hugging Face](https://huggingface.co/google/t5-v1_1-xxl)

See the `README.md` files in each directory for detailed download instructions.

### Step 2: Prepare 3D Point Cloud Features

Place your pre-extracted 3D point cloud feature files in the `3d_features/` directory:

- **Format**: `.pt` or `.npy` files
- **Naming**: `<sample_id>.pt` (e.g., `1.pt`, `2.pt`)
- **Content**: Each file should contain a 3D model feature vector (dimension: 64)

> **Note**: This implementation directly uses pre-extracted 3D features without scattering calculations. Feature files should contain geometric features extracted from 3D models.

### Step 3: Training

```powershell
./train_sd35m_24GLora.ps1
```

Training outputs will be saved in the `output/` directory:
- LoRA weights: `sd35m-test-lora-1e-4_*.safetensors`
- Fusion configuration: `sd35m-test-lora-1e-4_fusion.json`

### Step 4: Inference

#### Single Image Generation

```bash
python sd3_3d_lora_inference.py \
  --pretrained_model ./Stable-diffusion/SD3/sd3.5_medium.safetensors \
  --lora_weights ./output/sd35m-test-lora-1e-4_*.safetensors \
  --clip_l ./clip/clip_l.safetensors \
  --clip_g ./clip/clip_g.safetensors \
  --t5xxl ./clip/t5xxl_fp16.safetensors \
  --enable_3d_fusion \
  --pointcloud_dir ./3d_features \
  --fusion_method hybrid_film_attn \
  --prompt "your prompt here" \
  --pointcloud_path ./3d_features/1.pt \
  --output_dir ./output/inference
```

## 📁 Project Structure

```
for_github/
├── train_sd35m_24GLora.ps1      # Training script
├── sd3_3d_lora_inference.py     # Inference script
├── 3d_feature_fusion_v2.py      # Core tri-modal fusion module
├── requirements.txt             # Python dependencies
│
├── sd-scripts/                  # Training script library
│   ├── sd3_train_network_with_3d.py
│   ├── library/
│   │   ├── strategy_sd3_with_3d.py  # SD3 tri-modal fusion strategy
│   │   └── sd3_train_utils.py       # SD3 training utilities
│   └── networks/
│       └── lora_sd3.py              # SD3 LoRA network
│
├── sd3_lora_generation_code/    # Inference workflow code
│
├── train/                       # Training dataset
│   └── data/train/aircraft/     # Example dataset (10 samples)
│       ├── <id>.png             # Image files
│       ├── <id>.txt             # Text annotations
│       └── <id>_*_sd3.npz       # Image latent features
│
├── Stable-diffusion/SD3/        # SD3 model weights (download required)
├── clip/                        # Text encoder weights (download required)
├── VAE/                         # VAE weights (optional)
├── 3d_features/                 # 3D point cloud features directory
│   ├── 1.pt                     # 3D features for sample 1
│   ├── 2.pt                     # 3D features for sample 2
│   └── ...
│
├── output/                      # Output directory
│   ├── sd35m-test-lora-*.safetensors  # Trained LoRA weights
│   ├── *_fusion.json            # Fusion configurations
│   └── inference/               # Inference output images
│
├── logs/                        # Training logs
└── toml/                        # Configuration files
    └── qinglong.txt             # Example prompt file
```

## 🔬 Tri-Modal Feature Fusion

### Fusion Architecture

Our implementation fuses features from three modalities:

1. **Text Features**: Extracted via CLIP-L, CLIP-G, and T5-XXL encoders
2. **3D Point Cloud Features**: Directly uses pre-extracted 3D model features (no scattering computation required)
3. **Image Features**: Features extracted from image latent space

### Supported Fusion Methods

| Method | Description | Characteristics |
|--------|-------------|-----------------|
| `concat` | Concatenation fusion | Simple and direct, concatenates three features then projects |
| `add` | Addition fusion | Requires all modalities to have the same dimension |
| `attention` | Attention fusion | Uses multi-head attention to learn inter-modal relationships |
| `gentle` | Gentle fusion | Primarily preserves text features, lightly incorporates 3D information |
| `gated` | Gated fusion | Adaptively learns when to use 3D information |
| `attention_gated` | Attention-gated fusion | Combines attention and gating mechanisms |
| `image_guided` | Image-guided fusion | Pushes toward image semantics while maintaining fidelity |
| `hybrid_film_attn` | Hybrid FiLM+Attention fusion ⭐ | **Recommended method**, combines FiLM modulation and attention |

### Recommended Fusion Method

We recommend using `hybrid_film_attn`, which achieves high-quality fusion through:

1. **Multi-modal Gating**: Uses softmax weights to adaptively balance three modalities
2. **FiLM Modulation**: Generates per-channel gamma and beta based on fused features and image features
3. **Attention Refinement**: Further refines features through self-attention mechanism
4. **Image Semantic Push**: Pushes toward image semantics while maintaining generation fidelity

## ⚙️ Configuration

### Training Configuration

Edit key parameters in `train_sd35m_24GLora.ps1`:

```powershell
# Tri-modal feature fusion settings
$enable_3d_fusion = 1                   # Enable tri-modal feature fusion
$pointcloud_dir = "./3d_features"      # 3D point cloud feature directory
$fusion_method = "hybrid_film_attn"    # Fusion method
$pointcloud_dim = 64                   # 3D point cloud feature dimension
$image_feature_dim = 16                # Image feature dimension

# Model paths
$pretrained_model = "./Stable-diffusion/SD3/sd3.5_medium.safetensors"
$clip_l = "./clip/clip_l.safetensors"
$clip_g = "./clip/clip_g.safetensors"
$t5xxl = "./clip/t5xxl_fp16.safetensors"

# Training parameters
$resolution = "256,256"                 # Image resolution
$batch_size = 12                       # Batch size
$max_train_epoches = 100               # Maximum training epochs
$network_dim = 32                      # LoRA dimension
$network_alpha = 16                    # LoRA alpha
```

### Inference Configuration

Command-line arguments supported by the inference script:

```bash
# Required arguments
--pretrained_model <path>              # SD3 base model path
--clip_l <path>                        # CLIP-L encoder path
--clip_g <path>                        # CLIP-G encoder path
--t5xxl <path>                         # T5-XXL encoder path

# Tri-modal fusion arguments
--enable_3d_fusion                     # Enable 3D fusion
--pointcloud_dir <path>                # Point cloud feature directory
--fusion_method <method>               # Fusion method
--pointcloud_path <path>               # Specify point cloud feature file path

# Generation arguments
--prompt <text>                        # Text prompt
--prompts_file <path>                  # Batch prompt file
--height <int>                         # Image height (default: 512)
--width <int>                          # Image width (default: 512)
--sample_steps <int>                   # Sampling steps (default: 30)
--guidance_scale <float>               # Guidance scale (default: 7.5)
--seed <int>                           # Random seed
```

## 🔍 Technical Details

### 3D Feature Processing

This implementation **directly uses pre-extracted 3D model features** without scattering calculations. The feature processing pipeline:

1. **Feature Loading**: Loads 3D features from `.pt` or `.npy` files
2. **Feature Projection**: Projects 3D features to text feature dimension (768-dim) via linear layers
3. **Position Encoding**: Adds position encoding to distinguish different points
4. **Attention Processing**: Processes point cloud sequences using multi-head self-attention
5. **Global Pooling**: Obtains fixed-dimension feature vectors via global average pooling

### Image Feature Processing

Image features are extracted from latent space:

1. **Feature Loading**: Loads image latent features from `.npz` files
2. **Convolutional Processing**: Extracts spatial features via convolutional layers
3. **Global Pooling**: Extracts global image features
4. **Feature Projection**: Projects to the same dimension as text features

### Fusion Pipeline

The overall tri-modal fusion pipeline:

```
Text Features (CLIP-L/G + T5-XXL)
    ↓
3D Point Cloud Features (pre-extracted) ──→ [Feature Projection] ──→ [Fusion Module] ──→ Fused Features ──→ SD3.5 UNet
    ↓
Image Features (Latent Space) ──→ [Feature Projection] ──→
```

## ⚠️ Important Notes

1. **3D Feature Format**: 
   - This implementation directly uses pre-extracted 3D model features without scattering calculations
   - Feature files should be in `.pt` or `.npy` format
   - Feature dimension should match the `pointcloud_dim` parameter (default: 64)

2. **Path Configuration**: 
   - All paths are relative paths; ensure running from the `for_github` directory
   - Training and inference scripts automatically locate related files

3. **Model Weights**: 
   - All required weight files must be downloaded to run
   - Recommended to run `python check_weights.py` to verify file integrity

4. **Dataset**: 
   - Currently includes 10 samples for demonstration
   - More data is needed for actual training to achieve better results

## 🐛 Frequently Asked Questions

**Q: Training fails with "3D feature file not found"**  
A: Ensure that corresponding `.pt` or `.npy` files exist in the `3d_features/` directory, and filenames match dataset sample IDs (e.g., `1.pt`).

**Q: How to change the fusion method?**  
A: Modify the `$fusion_method` parameter in the training script, or use the `--fusion_method` argument during inference.

**Q: How to disable 3D fusion during inference?**  
A: Simply remove the `--enable_3d_fusion` argument; the model will use only text and image features.

**Q: How to adjust fusion strength?**  
A: Control fusion strength via the `FUSION_INJECTION_SCALE` environment variable (0.0-1.0, default: 1.0).

## 📄 Citation

If you find this project helpful for your research, please cite the GeoDiff-SAR paper

## 📝 License

Please refer to the original project's license.

## 🙏 Acknowledgments

- Stable Diffusion 3.5 model provided by [Stability AI](https://stability.ai/)
- Training framework based on [sd-scripts](https://github.com/kohya-ss/sd-scripts) project
- Thanks to all contributors for their support

---

**Note**: This implementation directly uses pre-extracted 3D model features without scattering calculations, simplifying the feature extraction process while maintaining high-quality SAR image generation capabilities.
