# MLV_IR_OD

checkpoint: 
```
mkdir checkpoints
wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10x.pt -O checkpoints/yolov10x.pt
```


## Install
Installation process follows yolov10
```
conda create -n mlv-ir-od python=3.11 -y
conda activate mlv-ir-od
pip install -r requirements.txt
pip install -e .
```