from __future__ import print_function
import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F

import utils


def train_one_epoch(model, criterion, optimizer, data_loader, epoch, val_dataloader, classes):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    running_corrects = 0
    epoch_data_len = len(data_loader.dataset)

    for i, (image, target) in enumerate(data_loader):
        batch_start = time.time()
        image, target = image.cuda(), target.cuda()
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(output, 1)

        loss_ = loss.item() * image.size(0) # this batch loss
        correct_ = torch.sum(preds == target.data) # this batch correct number

        running_loss += loss_
        running_corrects += correct_

        batch_end = time.time()
        if i % args.print_freq == 0 and i != 0:
            print('[TRAIN] Epoch: {}/{}, Batch: {}/{}, BatchAcc: {:.4f}, BatchLoss: {:.4f}, BatchTime: {:.4f}'.format(epoch,
                  args.epochs, i, epoch_data_len/args.batch_size, correct_.double()/image.size(0),
                  loss_/image.size(0), batch_end-batch_start))

        # if this result is the best, save it
        # show the best model in validation
        if i % args.eval_freq == 0 and i != 0:
            val_acc = evaluate(model, criterion, val_dataloader, epoch, i)
            model.train()
            # the first or best will save
            if len(g_val_accs) == 0 or val_acc > g_val_accs.get(max(g_val_accs, key=g_val_accs.get), 0.0):
                print('*** GET BETTER RESULT READY SAVE ***')
                if args.checkpoints:
                    torch.save({
                        'model': model.state_dict(),
                        'classes': classes,
                        'args': args},
                        os.path.join(args.checkpoints, 'model_{}_{}.pth'.format(epoch, i)))
                    print('*** SAVE.DONE. VAL_BEST_INDEX: {}_{}, VAL_BEST_ACC: {} ***'.format(epoch, i, val_acc))
            g_val_accs[str(epoch)+'_'+str(i)] = val_acc
            k = max(g_val_accs, key=g_val_accs.get)
            print('val_best_index: [ {} ], val_best_acc: [ {} ]'.format(k, g_val_accs[k]))

    lr=optimizer.param_groups[0]["lr"]
    epoch_loss = running_loss / epoch_data_len
    epoch_acc = running_corrects.double() / epoch_data_len
    epoch_end = time.time()
    print('[Train@] Epoch: {}/{}, EpochAcc: {:.4f}, EpochLoss: {:.4f}, EpochTime: {:.4f}, lr: {}'.format(epoch,
          args.epochs, epoch_acc, epoch_loss, epoch_end-epoch_start, lr))
    print()
    print()


def evaluate(model, criterion, data_loader, epoch, step):
    epoch_start = time.time()
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    epoch_data_len = len(data_loader.dataset)

    with torch.no_grad():
        for i, (image, target) in enumerate(data_loader):
            batch_start = time.time()
            image, target = image.cuda(), target.cuda()
            output = model(image)
            loss = criterion(output, target)

            _, preds = torch.max(output, 1)

            loss_ = loss.item() * image.size(0) # this batch loss
            correct_ = torch.sum(preds == target.data) # this batch correct number

            running_loss += loss_
            running_corrects += correct_

            batch_end = time.time()
            if i % args.print_freq == 0:
                print('[VAL] Epoch: {}/{}/{}, Batch: {}/{}, BatchAcc: {:.4f}, BatchLoss: {:.4f}, BatchTime: {:.4f}'.format(step,
                      epoch, args.epochs, i, epoch_data_len/args.batch_size, correct_.double()/image.size(0),
                      loss_/image.size(0), batch_end-batch_start))

        epoch_loss = running_loss / epoch_data_len
        epoch_acc = running_corrects.double() / epoch_data_len
        epoch_end = time.time()
        print('[Val@] Epoch: {}/{}, EpochAcc: {:.4f}, EpochLoss: {:.4f}, EpochTime: {:.4f}'.format(epoch,
              args.epochs, epoch_acc, epoch_loss, epoch_end-epoch_start))
        print()
    return epoch_acc


def main(args):
    print("Loading data")
    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    print("Loading training data")
    st = time.time()
    # need data augumentation
    dataset = torchvision.datasets.ImageFolder(
             traindir,
             transforms.Compose([
                    transforms.Resize((256, 256)),
                    #transforms.RandomResizedCrop(224),
                    transforms.RandomCrop(224),
                    transforms.RandomRotation(30),
                    #transforms.RandomGrayscale(p=0.4),
                    #transforms.Grayscale(num_output_channels=3),
                    #transforms.RandomAffine(45, shear=0.2),
                    #transforms.ColorJitter(),
                    transforms.RandomHorizontalFlip(),
                    #transforms.Lambda(utils.randomColor),
                    #transforms.Lambda(utils.randomBlur),
                    #transforms.Lambda(utils.randomGaussian),
                    transforms.ToTensor(),
                    normalize,]))

    print("Loading validation data")
    dataset_test = torchvision.datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize((224, 224)),
                    #transforms.CenterCrop(299),
                    #transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    normalize,]))

    print("Creating data loaders")
    data_loader = torch.utils.data.DataLoader(
                    dataset, batch_size=args.batch_size,
                    shuffle=True, num_workers=args.workers, pin_memory=True)

    # show all classes
    classes = data_loader.dataset.classes
    print(classes)

    val_dataloader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    model = torchvision.models.__dict__[args.model](pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    #model.fc = FC(num_ftrs, len(classes))
    print(model)

    # support muti gpu
    model = nn.DataParallel(model, device_ids=args.device)
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 80], gamma=args.lr_gamma)


    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    if args.test_only:
        evaluate(model, criterion, data_loader_test)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, optimizer, data_loader, epoch, val_dataloader, classes)
        lr_scheduler.step()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-dir', default='/data/user/yangfg/corpus/kar-data', help='dataset')
    parser.add_argument('--model', default='resnet101', help='model')
    parser.add_argument('--device', default=[0], help='device')
    parser.add_argument('-b', '--batch-size', default=512, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--eval-freq', default=50, type=int, help='validation frequency of batchs')
    parser.add_argument('--checkpoints', default='./checkpoints', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    args = parser.parse_args()

    if not os.path.exists(args.checkpoints):
        os.mkdir(args.checkpoints)

    g_val_accs = {}

    print(args)
    main(args)
