# -*- coding:utf-8 -*-
__author__ = 'tonye'

import torch
from PIL import Image
import random
import collections
import shutil
import time
import glob
import csv
import numpy as np
import os
import torch.nn as nn
import torch.utils.data as data



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1**(epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


def accuracy(y_pred, y_actual, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

# Helper Functions for Training
def train(train_loader, model, criterion, optimizer, epoch, cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if cuda:
            target = target.cuda(async=True)
        image_var = torch.autograd.Variable(images)
        label_var = torch.autograd.Variable(target)

        # compute y_pred
        y_pred = model(image_var)
        loss = criterion(y_pred, label_var)

        # measure accuracy and record loss
        prec1, prec1 = accuracy(y_pred.data, target, topk=(1, 1))
        losses.update(loss.data, images.size(0))
        acc.update(prec1, images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('   * EPOCH {epoch} {step}| Accuracy: {acc.avg:.3f} | Loss: {losses.avg:.3f}'.
              format(epoch=epoch, step=i, acc=acc, losses=losses))



def shear(img):
    width, height = img.size
    m = random.uniform(-0.05, 0.05)
    xshift = abs(m) * width
    new_width = width + int(round(xshift))
    img = img.transform((new_width, height), Image.AFFINE,
                        (1, m, -xshift if m > 0 else 0, 0, 1, 0),
                        Image.BICUBIC)
    return img



def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, labels) in enumerate(val_loader):
        labels = labels.cuda(async=True)
        image_var = torch.autograd.Variable(images, volatile=True)
        label_var = torch.autograd.Variable(labels, volatile=True)

        # compute y_pred
        y_pred = model(image_var)
        loss = criterion(y_pred, label_var)

        # measure accuracy and record loss
        prec1, temp_var = accuracy(y_pred.data, labels, topk=(1, 1))
        losses.update(loss.data[0], images.size(0))
        acc.update(prec1[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('#### EPOCH {epoch} | Accuracy: {acc.avg:.3f} | Loss: {losses.avg:.3f}'.format(epoch=epoch,
                                                                                         acc=acc,
                                                                                         losses=losses))

    return acc.avg


def test(test_loader, model, nb_aug, clip, submission_path, archs, epochs, nb_runs):
    csv_map = collections.defaultdict(float)

    # switch to evaluate mode
    model.eval()

    for aug in range(nb_aug):
        print("   * Predicting on test augmentation {}".format(aug + 1))

        for i, (images, filepath) in enumerate(test_loader):
            # pop extension, treat as id to map
            filepath = os.path.splitext(os.path.basename(filepath[0]))[0]
            filepath = int(filepath)

            image_var = torch.autograd.Variable(images, volatile=True)
            y_pred = model(image_var)
            # get the index of the max log-probability
            smax = nn.Softmax()
            smax_out = smax(y_pred)[0]
            cat_prob = smax_out.data[0]
            dog_prob = smax_out.data[1]
            prob = dog_prob
            if cat_prob > dog_prob:
                prob = 1 - cat_prob
            prob = np.around(prob, decimals=4)
            prob = np.clip(prob, clip, 1 - clip)
            csv_map[filepath] += (prob / nb_aug)

    sub_fn = submission_path + '{0}epoch_{1}clip_{2}runs'.format(epochs, clip, nb_runs)

    for arch in archs:
        sub_fn += "_{}".format(arch)

    print("Writing Predictions to CSV...")
    with open(sub_fn + '.csv', 'w') as csvfile:
        fieldnames = ['id', 'label']
        csv_w = csv.writer(csvfile)
        csv_w.writerow(('id', 'label'))
        for row in sorted(csv_map.items()):
            csv_w.writerow(row)
    print("Done.")


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class TestImageFolder(data.Dataset):
    def __init__(self, root, test_path, transform=None):
        images = []
        for filename in sorted(glob.glob(test_path + "*.jpg")):
            images.append('{}'.format(filename))

        self.root = root
        self.imgs = images
        self.transform = transform

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = Image.open(os.path.join(self.root, filename))
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)

def shear(img):
    width, height = img.size
    m = random.uniform(-0.05, 0.05)
    xshift = abs(m) * width
    new_width = width + int(round(xshift))
    img = img.transform((new_width, height), Image.AFFINE,
                        (1, m, -xshift if m > 0 else 0, 0, 1, 0),
                        Image.BICUBIC)
    return img