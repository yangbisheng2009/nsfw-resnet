#!/bin/bash

if [ "$1" = "train" ];
then
    nohup python train.py --model resnet101 --batch-size 64 --eval-freq 2000 --checkpoints checkpoints --data-dir /home/work/datasets/nsfw 1>log.log 2>err.log &
elif [ "$1" = "predict" ];
then
    python 1_predict_image.py --model resnet101 --checkpoint checkpoints/model_21_6000.pth --test-path /home/work/datasets/nsfw/test/porn/
elif [ "$1" = "test" ];
then
    python test_confusion_matrix.py --model resnet101 --data-dir /home/work/datasets/nsfw --checkpoint checkpoints/model_21_6000.pth
fi
