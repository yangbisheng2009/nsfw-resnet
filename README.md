<div align=center>
  <img width=400 src="https://github.com/yangbisheng2009/nsfw-resnet/blob/master/image/nsfw-logo.jpg" >
</div>

# NSFW

![Python 2.6](https://img.shields.io/badge/python-2.7-green.svg?style=plastic)
![Pytorch 0.4.0](https://img.shields.io/badge/pytorch-0.4.0-green.svg?style=plastic)
![cuDNN 7.3.1](https://img.shields.io/badge/cudnn-7.3.1-green.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-CC_BY--NC-green.svg?style=plastic)

## Description

Trained on 300,000 labled pictures:

- `porn` - pornography images
- `hentai` - hentai images, but also includes pornographic drawings
- `sexy` - sexually explicit images, but not pornography. Think nude photos, playboy, bikini, etc.
- `neutral` - safe for work neutral images of everyday things and people
- `drawings` - safe for work drawings (including anime)

## Requeriments

pytorch 0.4.0

## Usage

```shell
python train.py --model resnet101 --epochs 100

```

## Current status

## Detail

I have tried various methods include some pretrained models like resnet/inceptionv3 and data augumentation and finetuing.

Here are some tips which make a greate effect to the final result:

- data augumentation - make image rotaed,shifted,cropped,zoomed,flipped
- use pretrained model - use pretrained model by torchvision
- lock some layer and finetune FC - after train_init.py then lock some layer just finetune the FC
- adjust batch size - adjust the batch size meke it faster
- adjust learning rate - make lr dynamic when training in order to get saddle point

## Thanks
Thanks for my wife FeiFei Li. She gave me lots of encouragement. And made the beautiful logo for NSFW preject.  
Thanks for my workmate Kuai Li. He gave me lots of good suggestion.

## Join us
If you have good points.Join us!
You can attach me by:  
yangbisheng2009@gmail.com  
https://twitter.com/yangbisheng2009