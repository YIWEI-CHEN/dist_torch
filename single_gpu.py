import argparse
import glob
import logging
import sys
import time

import numpy as np
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
from torchvision.models import resnet50

import utils

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--epochs', type=int, default=400, help='num of training epochs')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.9, help='portion of training data')
parser.add_argument('--exec_script', type=str, default='scripts/eval.sh', help='script to run exp')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
if ',' in args.gpu:
    device_ids = list(int, map(args.gpu.split(',')))
else:
    device_ids = [int(args.gpu)]


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    torch.manual_seed(seed)
    cudnn.enabled = True
    cudnn.deterministic = True
    torch.cuda.manual_seed(seed)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    fix_seed(seed=args.seed)
    logging.info('gpu device = {}'.format(args.gpu))
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device_ids[0])
    model = resnet50(pretrained=False, progress=True, num_classes=CIFAR_CLASSES)

    model = model.to(device_ids[0])
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, test_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=0)

    test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)
    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    max_valid_acc = 0
    select_valid_obj = 0
    select_train_acc = 0
    select_train_obj = 0
    select_test_acc = 0
    select_test_obj = 0
    max_epoch = 0

    best_model_path = os.path.join(args.save, 'best_weights.pt')
    model_path = os.path.join(args.save, 'weights.pt')

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        # training
        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc %f, train_obj %f', train_acc, train_obj)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f, valid_obj %f', valid_acc, valid_obj)

        # test
        test_acc, test_obj = infer(test_queue, model, criterion)
        logging.info('test_acc %f, test_obj %f', test_acc, test_obj)

        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
            select_valid_obj = valid_obj
            select_train_acc = train_acc
            select_train_obj = train_obj
            select_test_acc = test_acc
            select_test_obj = test_obj
            utils.save(model, best_model_path)

        if os.path.exists(best_model_path):
            utils.copy(best_model_path, model_path)
        else:
            utils.save(model_path)

    logging.info('Best valid_acc {} (valid_obj {}) at Epoch {}'.format(max_valid_acc, select_valid_obj, max_epoch))
    logging.info('Corresponding train_acc {} (train_obj {})'.format(select_train_acc, select_train_obj))
    logging.info('Corresponding test_acc {} (test_obj {})'.format(select_test_acc, select_test_obj))


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (_input, target) in enumerate(train_queue):
        model.train()
        n = _input.size(0)

        _input = _input.to(device_ids[0])
        target = target.to(device_ids[0], non_blocking=True)

        optimizer.zero_grad()
        logits = model(_input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (_input, target) in enumerate(queue):
        _input = _input.to(device_ids[0])
        target = target.to(device_ids[0], non_blocking=True)

        logits = model(_input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = _input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()

