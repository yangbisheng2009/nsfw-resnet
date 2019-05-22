#nohup python train.py --model resnet101 --batch-size 512 --eval-freq 200 --output-dir checkpoint/resnet101 1>log.log 2>err.log &
nohup python predict.py --test-path /data/user/yangfg/experiment/nsfw/data/online-image --checkpoint checkpoint/resnet101/model_51_200.pth --model resnet101 > a &
