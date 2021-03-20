# CME
## Introduction
The code includes training and inference procedures for **Beyond Max-Margin: Class Margin Equilibrium for Few-shot Object Detection(CVPR2021).
Our code is based on https://github.com/bingykang/Fewshot_Detection and developed with Python 3.6.5 & PyTorch 1.1.0.
## Training our model on VOC
### Dataset preparation
If you want to train our model on VOC pascal dataset, please prepare dataset according to https://github.com/bingykang/Fewshot_Detection
### Training model on VOC split1
If you finish preparing dataset,
Please modify the dir in train_decoupling_disturbance.py and valid_decoupling.py
```
sys.path.append("Your Project dir")
```

After that,you can 
```
bash train_model.sh
```
to train model and get corresponding results on split1.
If you want to train other split, you only need to change
```
SPLIT=(1)
```
to which split you want to train.

You can download the pretrained weight by [Google Drive](https://drive.google.com/file/d/1-0Q2EqYXXb0dDm1J0e4dgyXeoQAesibU/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1WUrF0-dMyaS3InObBQa5zw ) with code: **CVPR** 

## Citing

```
@InProceedings{Li_2021_CVPR,
  author = {Li, Bohao and Yang, Boyu and Liu, Chang and Liu, Feng and Ji Rongrong and Ye, Qixiang},
  title = {Beyond Max-Margin: Class Margin Equilibrium for Few-shot Object Detection},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2021}
}
```
