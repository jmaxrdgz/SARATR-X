# Install & Setup
## On MacOS Silicon
```
conda create -n saratrx python=3.8
conda activate saratrx
pip install -r requirements.txt
```
Additionally you can check torch is working with Metal (mps) backend with :  
```python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"```  

# Dataset 
SAR images must be preprocessed as single precision tiles (.npy) before training.
The following command allows to chip images from a given path into 512x512 chips:  
```python data/chip_capella.py /path/to/sar_images --chip_size 512``` 

# Launch Training
```python train.py```