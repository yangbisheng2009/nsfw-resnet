<center>
<img src="https://github.com/yangbisheng2009/nsfw-resnet/blob/master/image/nsfw-detector.jpeg" alt="NSFW logo" width="300" />
</center>

# NSFW

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
python train.py --model resnet50 --epochs 100

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