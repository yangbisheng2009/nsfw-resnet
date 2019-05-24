from __future__ import print_function
import datetime
import os
import time
import sys
import traceback

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np

from utils import image_loader_url


def main(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformation = transforms.Compose([
                        transforms.Resize((224, 224)),
                        #transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,])
    '''

    transformation = transforms.Compose([
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        normalize,])
    '''

    classes = torch.load(args.checkpoint)['classes']
    model = torchvision.models.__dict__[args.model](pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))

    model = nn.DataParallel(model, device_ids=args.device)
    model.cuda()
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model.eval()

    count = 0

    for url in sys.stdin:
        count += 1
        try:
            url = url.strip()
            image_tensor = image_loader_url(url, transformation)
            image_tensor = image_tensor.float().unsqueeze_(0)
            input = image_tensor.cuda()
            output = model(input)
            index = output.data.cpu().numpy().argmax()
            label = classes[index]
            if label in ['hentai', 'sexy', 'porn']:
                print('{}\t{}\t{}'.format(url, classes[index], count))
                sys.stdout.flush()
        except:
            traceback.print_exc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--test-path', default='/data/user/yangfg/experiment/nsfw/data/online-image', help='dataset')
    parser.add_argument('--model', default='resnet101', help='model')
    parser.add_argument('--device', default=[0,1,2,3], help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--checkpoint', default='./checkpoint/101_without_data_aug/model_4_400.pth', help='checkpoint')

    args = parser.parse_args()

    print(args)
    main(args)
