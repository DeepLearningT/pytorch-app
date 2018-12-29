# -*- coding:utf-8 -*-
__author__ = 'tonye'


import os.path

import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


from HelperFunctions import *

ROOT_DIR = os.getcwd()
DATA_HOME_DIR = ROOT_DIR + '/data/DogsVsCats'
print DATA_HOME_DIR

# Config & Hyperparameters
# paths
data_path = DATA_HOME_DIR
split_train_path = data_path + '/train'
full_train_path = data_path + '/train_full/'
valid_path = data_path + '/valid/'
test_path = DATA_HOME_DIR + '/test'
saved_model_path = ROOT_DIR + '/models/'
submission_path = ROOT_DIR + '/submissions/'

# data
batch_size = 16
nb_split_train_samples = 23000
nb_full_train_samples = 25000
nb_valid_samples = 2000
nb_test_samples = 12500

# model
nb_runs = 1
nb_aug = 3
epochs = 35
lr = 1e-4
clip = 0.001
archs = ["resnet152"]
cuda = False

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))
best_prec1 = 0
print model_names


def main(mode="train", resume=False):
    global best_prec1

    for arch in archs:

        # create model
        print("=> Starting {0} on '{1}' model".format(mode, arch))
        model = models.__dict__[arch](pretrained=True)
        # Don't update non-classifier learned features in the pretrained networks
        for param in model.parameters():
            param.requires_grad = False
        # Replace the last fully-connected layer
        # Parameters of newly constructed modules have requires_grad=True by default
        # Final dense layer needs to replaced with the previous out chans, and number of classes
        # in this case -- resnet 101 - it's 2048 with two classes (cats and dogs)
        model.fc = nn.Linear(2048, 2)

        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
        else:
            model = torch.nn.DataParallel(model)



        # optionally resume from a checkpoint
        if resume:
            if os.path.isfile(resume):
                print("=> Loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume)
                start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
            else:
                print("=> No checkpoint found at '{}'".format(resume))

        if cuda:
            model = model.cuda()
            cudnn.benchmark = True
        else:
            cudnn.benchmark = False

        # Data loading code
        traindir = split_train_path
        valdir = valid_path
        testdir = test_path

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_loader = data.DataLoader(
            datasets.ImageFolder(traindir,
                                 transforms.Compose([
#                                      transforms.Lambda(shear),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalize,
                                 ])),
            batch_size=batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=True)

        val_loader = data.DataLoader(
            datasets.ImageFolder(valdir,
                                 transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalize,
                                 ])),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True)

        test_loader = data.DataLoader(
            TestImageFolder(testdir, test_path,
                            transforms.Compose([
#                                 transforms.Lambda(shear),
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                            ])),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False)

        if mode == "test":
            test(test_loader, model, nb_aug, clip, submission_path, archs, epochs, nb_runs)
            return

        # define loss function (criterion) and pptimizer
        criterion = nn.CrossEntropyLoss()
        if cuda:
            criterion = criterion.cuda()

        if mode == "validate":
            validate(val_loader, model, criterion, 0)
            return

        global lr
        optimizer = optim.Adam(model.module.fc.parameters(), lr, weight_decay=1e-4)

        for epoch in range(epochs):
            adjust_learning_rate(optimizer, epoch, lr)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, cuda)

            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion, epoch)

            # remember best Accuracy and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)


if __name__ == '__main__':
    main(mode="train")