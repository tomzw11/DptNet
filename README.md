# DptNet
DPTNet in mindspore
Dual-Path Transformer Network: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation

pip install -r requirements.txt

python preprocess.py

python train.py [DEVICE_ID]

python eval.py
参数:
model_path                 ckpt文件
data-dir                   测试集路径
batch_size                 测试集batch大小
