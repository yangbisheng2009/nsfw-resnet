#!/bin/bash

if [ "$1" = "train" ];
then
    python train.py --model resnet101 --batch-size 64 --eval-freq 2000 --checkpoints checkpoints
elif [ "$1" = "predict" ];
then
    python 1_predict_image.py --model resnet101 --checkpoint checkpoints/model_21_6000.pth --test-path inputs/
elif [ "$1" = "test" ];
then
    python test_confusion_matrix.py --model resnet101
fi
