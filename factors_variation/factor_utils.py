import os
import sys
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

import kornia.augmentation as A

from models import vgg11, vgg13, resnet18, resnet34, NiN, densenet_cifar, MobileNetV2, WideResNet, PreActResNet34
from GTSRB import *
from factor_backdoors import *


EPSILON = 1e-7

_dataset_name = ['cifar10', 'gtsrb', 'stl10', 'cifar100']

_mean = {
    'cifar10':  [0.4914, 0.4822, 0.4465],
    'stl10':    [0.4409, 0.4274, 0.3849],
    'gtsrb':    [0.3337, 0.3064, 0.3171],
    'cifar100': [0.4802, 0.4481, 0.3975],
}

_std = {
    'cifar10':  [0.2023, 0.1994, 0.2010],
    'stl10':    [0.2603, 0.2566, 0.2713],
    'gtsrb':    [0.2672, 0.2564, 0.2629],
    'cifar100': [0.2675, 0.2565, 0.2761],
}

_size = {
    'cifar10':  (32, 32),
    'stl10':    (32, 32),
    'gtsrb':    (32, 32),
    'cifar100': (32, 32),
}

_num = {
    'cifar10':  10,
    'stl10':    10,
    'gtsrb':    43,
    'cifar100': 100,
}


def get_config(dataset):
    assert dataset in _dataset_name, _dataset_name
    config = {}
    config['mean'] = _mean[dataset]
    config['std']  = _std[dataset]
    config['size'] = _size[dataset]
    config['num_classes'] = _num[dataset]
    return config


def get_norm(dataset):
    assert dataset in _dataset_name, _dataset_name
    mean = torch.FloatTensor(_mean[dataset])
    std  = torch.FloatTensor(_std[dataset])
    normalize   = transforms.Normalize(mean, std)
    unnormalize = transforms.Normalize(- mean / std, 1 / std)
    return normalize, unnormalize


def get_transform(args, augment=False, tensor=False):

    transforms_list = []
    if augment:
        transforms_list.append(transforms.Resize(_size[args.dataset]))
        transforms_list.append(transforms.RandomCrop(_size[args.dataset], padding=4))
        
        # Horizontal Flip
        transforms_list.append(transforms.RandomHorizontalFlip())
    else:
        transforms_list.append(transforms.Resize(_size[args.dataset]))
    
    # To Tensor
    if not tensor:
        transforms_list.append(transforms.ToTensor())

    transform = transforms.Compose(transforms_list)
    return transform


def get_dataset(args, train=True, augment=True):
    transform = get_transform(args, augment=train & augment)
    
    if args.dataset == 'cifar10':
        dataset = datasets.CIFAR10(args.datadir, train, download=False, transform=transform)
    elif args.dataset == 'stl10':
        dataset = datasets.STL10(args.datadir, split='train' if train else 'test', download=False, transform=transform)
    elif args.dataset == 'gtsrb':
        split = 'train' if train else 'test'
        dataset = GTSRB(args.datadir, split, transform, download=False)
    elif args.dataset == 'cifar100':
        dataset = datasets.CIFAR100(args.datadir, train, download=False, transform=transform)

    return dataset


def get_model(args):
    num_classes = _num[args.dataset]

    # Nomral vgg, resnet
    if args.network == 'resnet18':
        model = resnet18(num_classes=num_classes)
    elif args.network == 'resnet34':
        model = resnet34(num_classes=num_classes)
    elif args.network == 'vgg11':
        model = vgg11(num_classes=num_classes)
    elif args.network == 'vgg13':
        model = vgg13(num_classes=num_classes)
    elif args.network == 'prn34':
        model = PreActResNet34(num_classes=num_classes)
    elif args.network == 'wrn':
        model = WideResNet(num_classes=num_classes)
    elif args.network == 'densenet':
        model = densenet_cifar(num_classes=num_classes)
    elif args.network == 'mobilenet':
        model = MobileNetV2(num_classes=num_classes)
    else:
        raise NotImplementedError
    
    return model


# print configurations
def print_args(opt):
    message = ''
    message += '='*46 +' Options ' + '='*46 +'\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''

        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '='*48 +' End ' + '='*47 +'\n'
    print(message)


def get_backdoor(attack, side_len):
    if attack == 'badnet':
        backdoor = Badnet(side_len)
    elif attack == 'blend':
        backdoor = Blend(side_len)
    elif attack == 'wanet':
        backdoor = WaNet(side_len)
    else:
        raise NotImplementedError
    
    return backdoor


class PoisonDataset(Dataset):
    def __init__(self, dataset, attack, victim, target, poison_rate, backdoor=None):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.attack = attack
        self.victim = victim
        self.target = target

        self.n_data = len(self.dataset)
        self.n_poison = int(self.n_data * poison_rate)
        self.n_normal = self.n_data - self.n_poison

        if backdoor is None:
            self.backdoor = get_backdoor(self.attack, self.dataset[0][0].shape[-1])
        else:
            self.backdoor = backdoor

    def __getitem__(self, index):
        i = np.random.randint(0, self.n_data)
        img, lbl = self.dataset[i]
        if index < self.n_poison:
            # Universal attack
            if self.victim == -1:
                while lbl == self.target:
                    i = np.random.randint(0, self.n_data)
                    img, lbl = self.dataset[i]
                lbl = self.target
            # Label-specific attack
            else:
                while lbl != self.victim:
                    i = np.random.randint(0, self.n_data)
                    img, lbl = self.dataset[i]
                lbl = self.target
            
            img = self.inject_trigger(img)

        return img, lbl

    def __len__(self):
        return self.n_normal + self.n_poison

    def inject_trigger(self, img):
        img = img.unsqueeze(0)
        img = self.backdoor.inject(img)[0]
        return img


class ProbTransform(nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


class PostTensorTransform(nn.Module):
    def __init__(self, shape):
        super(PostTensorTransform, self).__init__()
        self.random_crop = ProbTransform(A.RandomCrop(shape, padding=4), p=0.5)
        self.random_rotation = ProbTransform(A.RandomRotation(10), p=0.5)
        self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
