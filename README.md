# Install & Setup
## On Linux/MacOS
```
conda create -n saratrx python=3.8
conda activate saratrx
pip install -r requirements.txt
```
Additionally you can check torch is working with Metal (for MacOS) backend with :  
```python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"``` 
  
Download [MAEHiViT Imagnet weights](https://drive.google.com/file/d/1VZQz4buhlepZ5akTcEvrA3a_nxsQZ8eQ/view) and add it to the project (training initialization).  
```
# Download MAE-HiViT pretrained weights
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1VZQz4buhlepZ5akTcEvrA3a_nxsQZ8eQ" -O mae_hivit_base_1600ep.pth

# Move it to the project weights folder
mkdir -p checkpoints/pretrained
mv mae_hivit_base_1600ep.pth checkpoints/pretrained/
```

# Dataset 
SAR images must be preprocessed as single precision tiles (.npy) before training.
The following command allows to chip images from a given path into 512x512 chips:  
```python data/chip_capella.py /path/to/sar_images --chip_size 512``` 

# Launch Training
```python train.py```