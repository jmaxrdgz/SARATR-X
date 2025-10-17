# Install & Setup
## On MacOS Silicon
```
conda create -n saratrx python=3.8
conda activate saratrx
pip install -r requirements.txt
```
Additionally you can check torch is working with Metal (mps) backend with :  
```python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"```  

# Launch Training
```python train.py```