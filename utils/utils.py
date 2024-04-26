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

from models import vgg11, vgg13, resnet18, resnet34, NiN, densenet_cifar, MobileNetV2, WideResNet, PreActResNet34, resnet18_nonlinear
from GTSRB import *
from backdoors import *


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


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


def get_backdoor(attack, shape, device=None):
    if attack == 'badnet':
        backdoor = Badnet(shape, device=device)
    elif attack == 'patch':
        # Pure-color patch trigger
        backdoor = Patch(shape, device=device)
    elif attack == 'trojnn':
        backdoor = TrojNN(shape, device=device)
    elif attack == 'dynamic':
        backdoor = Dynamic(device=device)
    elif attack == 'inputaware':
        backdoor = InputAware(device=device)
    elif attack == 'reflection':
        backdoor = Reflection(shape, device=device)
    elif attack == 'sig':
        backdoor = SIG(device=device)
    elif attack == 'blend':
        backdoor = Blend(device=device)
    elif attack == 'wanet':
        backdoor = WaNet(shape, device=device)
    elif attack == 'invisible':
        backdoor = Invisible(device=device)
    elif attack == 'filter':
        backdoor = Filter(device=device)
    elif attack == 'lira':
        backdoor = LIRA(device=device)
    elif attack == 'dfst':
        backdoor = DFST(device=device)
    elif attack == 'adaptive_blend':
        backdoor = AdaptiveBlend(device=device)
    elif attack == 'cl':
        backdoor = CleanLabel(shape, device=device)
    else:
        raise NotImplementedError
    
    return backdoor


class PoisonDataset(Dataset):
    def __init__(self, dataset, threat, attack, target, poison_rate, backdoor=None, victim=None):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.threat = threat
        self.attack = attack
        self.target = target
        self.victim = victim

        self.n_data = len(self.dataset)
        self.n_poison = int(self.n_data * poison_rate)
        self.n_normal = self.n_data - self.n_poison

        if backdoor is None:
            self.backdoor = get_backdoor(self.attack, self.dataset[0][0].shape[1:])
        else:
            self.backdoor = backdoor
        
        self.device = self.backdoor.device

    def __getitem__(self, index):
        i = np.random.randint(0, self.n_data)
        img, lbl = self.dataset[i]
        if index < self.n_poison:
            if self.threat.startswith('clean'):
                while lbl != self.target:
                    i = np.random.randint(0, self.n_data)
                    img, lbl = self.dataset[i]
            elif self.threat.startswith('dirty'):
                while lbl == self.target:
                    i = np.random.randint(0, self.n_data)
                    img, lbl = self.dataset[i]
                lbl = self.target
            elif self.threat.startswith('specific'):
                while lbl != self.victim:
                    i = np.random.randint(0, self.n_data)
                    img, lbl = self.dataset[i]
                lbl = self.target
            else:
                raise NotImplementedError
            
            img = self.inject_trigger(img)

        return img, lbl

    def __len__(self):
        return self.n_normal + self.n_poison

    def inject_trigger(self, img):
        img = img.unsqueeze(0).to(self.device)
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
        self.random_crop = ProbTransform(A.RandomCrop(shape, padding=5), p=0.8)
        self.random_rotation = ProbTransform(A.RandomRotation(10), p=0.5)
        self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class TargetDataset(Dataset):
    def __init__(self, dataset, target):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.target = target

        self.x = []
        self.y = []
        for img, lbl in self.dataset:
            if lbl == self.target:
                self.x.append(img)
                self.y.append(lbl)

    def __getitem__(self, index):
        img = self.x[index]
        lbl = self.y[index]

        return img, lbl

    def __len__(self):
        return len(self.x)


def get_poison_testset(args, test_set):
    if args.attack == 'composite':
        mixer = HalfMixer()
        ca, cb, cc = 0, 1, 2
        poison_set = MixDataset(dataset=test_set, mixer=mixer, classA=ca,
                                classB=cb, classC=cc, data_rate=1,
                                normal_rate=0, mix_rate=0,
                                poison_rate=1)
    elif args.attack in ['dfst', 'invisible']:
        poison_set = torch.load(os.path.join(args.datadir, f'{args.dataset}_{args.attack}.pt'))
        x_poison, y_poison = poison_set.tensors
        new_x_poison, new_y_poison = [], []
        for i in range(len(y_poison)):
            if y_poison[i] != args.target:
                new_x_poison.append(x_poison[i])
                new_y_poison.append(y_poison[i] * 0 + args.target)
        x_poison = torch.stack(new_x_poison)
        y_poison = torch.stack(new_y_poison)
        poison_set = torch.utils.data.TensorDataset(x_poison, y_poison)
    else:
        poison_set = PoisonDataset(dataset=test_set,
                                    threat='dirty',
                                    attack=args.attack,
                                    target=args.target,
                                    poison_rate=1)

        trigger_filepath = f'data/trigger/{args.attack}/{args.dataset}_{args.network}'
        if args.suffix != '':
            trigger_filepath += f'_{args.suffix}'

        if args.attack == 'inputaware':
            poison_set.backdoor.net_mask = torch.load(trigger_filepath + '_mask.pt', map_location='cpu')
            poison_set.backdoor.net_mask.eval()
            poison_set.backdoor.net_genr = torch.load(trigger_filepath + '_genr.pt', map_location='cpu')
            poison_set.backdoor.net_genr.eval()
        elif args.attack in ['dynamic', 'lira']:
            poison_set.backdoor.net_genr = torch.load(trigger_filepath + '_genr.pt', map_location='cpu')
            poison_set.backdoor.net_genr.eval()

    return poison_set


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--datadir', default='./data', help='root directory of data')

    parser.add_argument('--dataset', default='cifar10', help='dataset')
    parser.add_argument('--network', default='resnet18', help='network structure')
    parser.add_argument('--attack', default='clean', help='attack method')

    parser.add_argument('--suffix', default='', help='suffix of the model')

    parser.add_argument('--target', type=int, default=0, help='target label')
    parser.add_argument('--ratio', type=float, default=0.05, help='ratio of the dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')

    parser.add_argument('--seed', type=int, default=123, help='seed index')

    parser.add_argument('--scratch_dirpath', default='./scratch', help='scratch directory')

    args = parser.parse_args()

    import pickle
    dataset = get_dataset(args, train=False)
    x_val, y_val = [], []
    for x, y in dataset:
        x_val.append(x)
        y_val.append(y)
    
    x_val = torch.stack(x_val).permute(0, 2, 3, 1).numpy()
    # Convert to np.uint8
    x_val = (x_val * 255).astype(np.uint8)
    y_val = np.array(y_val)

    # Randomly select 100 images averagely
    num_classes = get_config(args.dataset)['num_classes']
    class_cnt = [0 for _ in range(num_classes)]
    n_per_cls = int(np.ceil(100 / num_classes))
    new_x_val, new_y_val = [], []
    for i in range(len(x_val)):
        if class_cnt[y_val[i]] < n_per_cls:
            new_x_val.append(x_val[i])
            new_y_val.append(y_val[i])
            class_cnt[y_val[i]] += 1
    x_val = np.stack(new_x_val)
    y_val = np.array(new_y_val)

    x_val, y_val = x_val[:100], y_val[:100]

    print(x_val.shape, y_val.shape)

    idx = 3
    img = Image.fromarray(x_val[idx])
    img.save('tmp.png')

    # Save
    data = {'x_val': x_val, 'y_val': y_val}
    with open('data/cifar10_detval_samples', 'wb') as f:
        pickle.dump(data, f)
