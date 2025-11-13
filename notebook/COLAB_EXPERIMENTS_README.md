# SARATR-X Google Colab Training Experiments

This directory contains a comprehensive Google Colab notebook for running SARATR-X training experiments with different reconstruction techniques on the Sentinel-1 & Sentinel-2 dataset.

## Notebook: `colab_training_experiments.ipynb`

### Overview

This notebook enables you to train the SARATR-X model (with MAE-HiViT tiny backbone) using 4 different reconstruction techniques, all optimized for Google Colab's free T4 GPU tier.

### Experiments

#### 1. Pixel-Reconstruction (SAR → SAR)
- **Input**: Sentinel-1 SAR images
- **Target**: Same SAR images
- **Purpose**: Self-supervised reconstruction to learn SAR-specific features
- **Target Mode**: `optical` (configured to use SAR as both input and target)

#### 2. MGF-Reconstruction (SAR → Multi-scale Gradient Features)
- **Input**: Sentinel-1 SAR images
- **Target**: Multi-scale gradient features extracted from SAR
- **Purpose**: Learn robust features by reconstructing gradient information at multiple scales
- **Target Mode**: `mgf`
- **Kernel Sizes**: [9, 13, 17]

#### 3. RGB-Reconstruction (SAR → RGB Optical)
- **Input**: Sentinel-1 SAR images
- **Target**: Sentinel-2 RGB optical images
- **Purpose**: Learn cross-modal translation from SAR to optical domain
- **Target Mode**: `optical`

#### 4. Greyscale-Reconstruction (SAR → Greyscale Optical)
- **Input**: Sentinel-1 SAR images
- **Target**: Greyscale-converted Sentinel-2 optical images
- **Purpose**: Simplified cross-modal learning without color information
- **Target Mode**: `optical` (with greyscale conversion)

### Features

#### Colab Free Tier Optimizations
- **GPU**: Optimized for T4 GPU
- **Batch Size**: 32 (reduced from 64 for memory constraints)
- **Epochs**: 50 (reduced from 200 for time constraints)
- **Mixed Precision**: 16-bit for faster training
- **Session Management**: Handles 1.5-hour free tier limit

#### Persistent Storage
All results are saved to **Google Drive** and persist after the Colab session ends:
- **Checkpoints**: Saved every 5 epochs
- **Logs**: TensorBoard logs for visualization
- **Best Models**: Top 3 checkpoints by validation loss

#### Dataset
- **Source**: Kaggle dataset `requiemonk/sentinel12-image-pairs-segregated-by-terrain`
- **Format**: Automatically converts PNG images to NPY format for faster loading
- **Size**: 256x256 pixel pairs
- **Types**: Multiple terrain types included

## Quick Start Guide

### Prerequisites

1. **Google Account** with Google Drive
2. **Kaggle Account** with API credentials
3. **Google Colab** (free tier is sufficient)

### Step-by-Step Instructions

#### 1. Open the Notebook in Google Colab

**Option A: Direct Upload**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click `File → Upload notebook`
3. Upload `colab_training_experiments.ipynb`

**Option B: From GitHub**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click `File → Open notebook → GitHub`
3. Enter: `https://github.com/jmaxrdgz/SARATR-X`
4. Select: `notebook/colab_training_experiments.ipynb`

#### 2. Get Kaggle API Credentials

1. Go to [https://www.kaggle.com/account](https://www.kaggle.com/account)
2. Scroll to the "API" section
3. Click "Create New API Token"
4. Download `kaggle.json` (save it for step 5)

#### 3. Set Runtime to GPU

1. In Colab: `Runtime → Change runtime type`
2. Select: `T4 GPU`
3. Click: `Save`

#### 4. Run Initial Setup Cells (1-4)

Execute cells sequentially:
- Cell 1: Mount Google Drive (authorize when prompted)
- Cell 2: Check GPU availability
- Cell 3: Clone repository
- Cell 4: Install dependencies

#### 5. Upload Kaggle Credentials (Cell 5)

When cell 5 runs, it will prompt you to upload `kaggle.json`:
1. Click the "Choose Files" button
2. Select the `kaggle.json` you downloaded earlier
3. Wait for confirmation

#### 6. Download and Preprocess Dataset (Cells 6-8)

Run cells 6-8 to:
- Download Sentinel-1&2 dataset (~5-10 minutes)
- Convert PNG images to NPY format (~2-3 minutes)
- Verify dataset integrity

#### 7. Create Experiment Configs and Training Script (Cells 9-10)

Run cells 9-10 to:
- Generate 4 experiment configuration files
- Create custom training script

#### 8. Run Experiments (Cells 11)

Run each experiment cell in section 11:
- Each experiment takes approximately 15-20 minutes on T4 GPU
- Checkpoints are saved every 5 epochs to Google Drive
- You can run all 4 experiments sequentially (~1-1.5 hours total)

**Important**: If your session expires before all experiments complete:
1. Restart the notebook
2. Re-run setup cells (1-4)
3. Skip data download cells (6-8) if already downloaded
4. Use cell 14 to resume from last checkpoint

#### 9. Monitor Training (Cell 12)

Run cell 12 to launch TensorBoard:
- View real-time training loss curves
- Compare experiments side-by-side
- TensorBoard stays active while training runs

#### 10. View Results (Cell 13)

After training completes, run cell 13 to:
- See summary of all experiments
- Check number of checkpoints saved
- View checkpoint file sizes

## File Structure After Running

```
Google Drive/
└── MyDrive/
    └── SARATRX_experiments/
        ├── checkpoints/
        │   ├── exp1_pixel_sar/
        │   │   ├── last.ckpt
        │   │   ├── epoch10-loss0.1234.ckpt
        │   │   └── ...
        │   ├── exp2_mgf_sar/
        │   ├── exp3_rgb_sar_to_rgb/
        │   └── exp4_grey_sar_to_grey/
        └── logs/
            ├── exp1_pixel_sar/
            │   └── version_0/
            │       └── events.out.tfevents...
            ├── exp2_mgf_sar/
            ├── exp3_rgb_sar_to_rgb/
            └── exp4_grey_sar_to_grey/
```

## Training Configuration

### Model Architecture
- **Backbone**: MAE-HiViT (Hierarchical Vision Transformer)
- **Encoder Depth**: [2, 2, 20]
- **Embed Dimension**: 512
- **Decoder Depth**: 6
- **Number of Heads**: 8 (encoder), 16 (decoder)
- **Mask Ratio**: 75%

### Training Hyperparameters
- **Optimizer**: Adam
- **Learning Rate**: 1.5e-4
- **Weight Decay**: 0.05
- **Beta1/Beta2**: 0.9/0.95
- **Warmup Epochs**: 5
- **Total Epochs**: 50 (per experiment)
- **Batch Size**: 32
- **Gradient Clipping**: 5.0

### Data Augmentation
- Random resized crop (scale: 0.2-1.0)
- Random horizontal flip
- Color jitter (contrast: 0.5)
- Paired transformations (SAR and optical receive identical transforms)

## Troubleshooting

### Session Expires Before Completion

**Solution**: Use the resume training feature (Cell 14)
1. Identify which experiment was interrupted
2. Run cell 14 to update the config with last checkpoint path
3. Re-run the experiment cell to continue training

### Out of Memory Error

**Solutions**:
1. Reduce batch size in cell 9:
   ```python
   base_config['train']['batch_size'] = 16  # Change from 32
   ```
2. Reduce image size:
   ```python
   base_config['data']['img_size'] = 128  # Change from 256
   ```

### Kaggle Dataset Download Fails

**Solutions**:
1. Verify `kaggle.json` is correctly uploaded
2. Check Kaggle API credentials are valid
3. Ensure you've accepted the dataset terms on Kaggle website
4. Try alternative download method (see download_sentinel_dataset.ipynb)

### TensorBoard Not Showing

**Solution**:
1. Make sure cell 12 runs without errors
2. Wait a few seconds for TensorBoard to load
3. Refresh the browser if needed
4. Check that logs directory exists in Google Drive

## Tips for Best Results

### Optimize GPU Usage
- Run experiments during off-peak hours for better GPU availability
- Close TensorBoard when not actively monitoring to save memory
- Use mixed precision training (already enabled)

### Maximize Training Time
- Train during periods when you can monitor the session
- Use browser extensions to keep Colab session active
- Consider Colab Pro for longer runtimes if needed

### Compare Experiments
- Use TensorBoard to compare all 4 experiments side-by-side
- Pay attention to convergence speed and final loss values
- RGB-reconstruction typically converges faster than MGF

### Save Important Results
- Download best checkpoints from Google Drive
- Export TensorBoard logs for offline analysis
- Take screenshots of key training curves

## Expected Training Times (T4 GPU)

| Experiment | Time per Epoch | Total Time (50 epochs) |
|------------|---------------|------------------------|
| Exp 1: Pixel-SAR | ~1.5 min | ~75 min |
| Exp 2: MGF-SAR | ~2 min | ~100 min |
| Exp 3: RGB-SAR-to-RGB | ~1.5 min | ~75 min |
| Exp 4: Greyscale | ~1.5 min | ~75 min |

**Note**: Times are approximate and may vary based on GPU availability and batch size.

## Next Steps After Training

### Evaluate Models
Use the saved checkpoints to:
1. Run inference on test images
2. Compute quantitative metrics (MSE, PSNR, SSIM)
3. Visualize reconstruction quality

### Fine-tune Best Model
1. Identify the best-performing experiment
2. Resume training for more epochs
3. Experiment with different hyperparameters

### Transfer to Other Datasets
Use the trained models as starting points for:
- Other SAR datasets
- Different reconstruction tasks
- Fine-tuning on specific terrain types

## Support

For issues or questions:
- Check the main repository README
- Review existing GitHub issues
- Create a new issue with details about your problem

## References

- **SARATR-X Paper**: [Link to paper]
- **HiViT Architecture**: [Link to paper]
- **Sentinel Dataset**: [Kaggle dataset page](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain)

---

**Happy Training! 🚀**
