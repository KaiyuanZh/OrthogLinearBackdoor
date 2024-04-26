import os
import sys
import time
import copy
import argparse
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from utils.utils import *
from utils.attack import *

import warnings
warnings.filterwarnings('ignore')


Print_level = 1


def seam(args, model, train_loader, test_loader, poison_test_loader, preprocess):
    total_time = 0
    time_start = time.time()

    # Step 1: Forgetting
    # Set forgetting lr=1e-2
    # lr = 3e-2
    lr = 1e-2
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    num_classes = get_config(args.dataset)['num_classes']
    acc_bound = 0.2
    # acc_bound = 0.1

    if args.dataset == 'cifar10':
        fg_rate = 1e-3
    elif args.dataset == 'gtsrb':
        fg_rate = 5e-3

    forget_set = get_dataset(args, train=True, augment=True)
    forget_set = FinetuneDataset(forget_set, num_classes=num_classes, data_rate=fg_rate)
    forget_loader = DataLoader(dataset=forget_set, batch_size=args.batch_size, shuffle=True)

    recover_set = get_dataset(args, train=True, augment=True)
    recover_set = FinetuneDataset(recover_set, num_classes=num_classes, data_rate=0.1)
    recover_loader = DataLoader(dataset=recover_set, batch_size=args.batch_size, shuffle=True)

    forget_step = 0
    while True:
        forget_step += 1
        model.train()
        acc = []
        for step, (x_batch, y_batch) in enumerate(forget_loader):
            # Randomly relabel y_batch
            y_forget = torch.randint(0, num_classes, y_batch.shape).cuda()

            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

            optimizer.zero_grad()

            output = model(preprocess(x_batch))
            loss = criterion(output, y_forget)
            loss.backward()
            optimizer.step()

            y_pred = torch.max(output.data, 1)[1]
            acc.append((y_pred == y_batch))

            if forget_step >= 100:
                break

        acc = torch.cat(acc, dim=0)
        n_forget = acc.shape[0]
        acc = acc.sum().item() / n_forget
        print(f'Forgetting --- Step: {forget_step}, ACC: {acc*100:.2f}% ({n_forget})')

        if acc < acc_bound:
            print('Forgot enough samples')
            break
    
    # Step 2: Recovering
    epochs = 10
    for epoch in range(epochs):
        model.train()
        if epoch > 0 and epoch % 2 == 0:
            lr /= 10

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for step, (x_batch, y_batch) in enumerate(recover_loader):
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

            optimizer.zero_grad()

            output = model(preprocess(x_batch))
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 2 == 0:
            time_end = time.time()

            model.eval()
            correct_cl = 0
            correct_bd = 0

            with torch.no_grad():
                total_cl = 0
                for (x_test, y_test) in test_loader:
                    x_test = x_test.cuda()
                    y_test = y_test.cuda()
                    total_cl += y_test.size(0)

                    ### clean accuracy ###
                    y_out = model(preprocess(x_test))
                    _, y_pred = torch.max(y_out.data, 1)
                    correct_cl += (y_pred == y_test).sum().item()
                
                total_bd = 0
                for (x_test, y_test) in poison_test_loader:
                    x_test = x_test.cuda()
                    y_test = y_test.cuda()
                    total_bd += y_test.size(0)

                    ### backdoor accuracy ###
                    y_out = model(preprocess(x_test))
                    _, y_pred = torch.max(y_out.data, 1)
                    correct_bd += (y_pred == y_test).sum().item()

            acc = correct_cl / total_cl
            asr = correct_bd / total_bd

            if Print_level > 0:
                sys.stdout.write('epoch: {:2}/{}, lr: {:.4f} - {:.2f}s, '
                                    .format(epoch+1, epochs, lr, time_end-time_start)\
                                    + 'loss: {:.4f}, acc: {:.4f}, asr: {:.4f}\n'
                                    .format(loss, acc, asr))
                sys.stdout.flush()

            total_time += (time_end-time_start)
            time_start = time.time()

    return model, total_time


##############################################################################
# Test
##############################################################################
def test(model, test_loader, poison_test_loader, preprocess):
    # Evaluate on the result model
    model.eval()

    correct_cl = 0
    correct_bd = 0

    with torch.no_grad():
        total_cl = 0
        for (x_test, y_test) in test_loader:
            x_test = x_test.cuda()
            y_test = y_test.cuda()
            total_cl += y_test.size(0)

            ### clean accuracy ###
            y_out = model(preprocess(x_test))
            _, y_pred = torch.max(y_out.data, 1)
            correct_cl += (y_pred == y_test).sum().item()
        
        total_bd = 0
        for (x_test, y_test) in poison_test_loader:
            x_test = x_test.cuda()
            y_test = y_test.cuda()
            total_bd += y_test.size(0)

            ### backdoor accuracy ###
            y_out = model(preprocess(x_test))
            _, y_pred = torch.max(y_out.data, 1)
            correct_bd += (y_pred == y_test).sum().item()

    acc = correct_cl / total_cl
    asr = correct_bd / total_bd

    return acc, asr


##############################################################################
# Main
##############################################################################
class FinetuneDataset(Dataset):
    def __init__(self, dataset, num_classes, data_rate=1):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset

        # Randomly select data_rate of the dataset
        n_data = len(dataset)
        n_single = int(n_data * data_rate / num_classes)
        self.n_data = n_single * num_classes

        # Evenly select data_rate of the dataset
        cnt = [n_single for _ in range(num_classes)]

        self.indices = np.random.choice(n_data, n_data, replace=False)

        self.data = []
        self.targets = []
        for i in self.indices:
            img, lbl = dataset[i]

            if cnt[lbl] > 0:
                self.data.append(img)
                self.targets.append(lbl)
                cnt[lbl] -= 1

    def __getitem__(self, index):
        img, lbl = self.data[index], self.targets[index]
        return img, lbl

    def __len__(self):
        return self.n_data


def main(args, preeval=True):
    # Load attacked model
    model_filepath = f'ckpt/{args.dataset}_{args.network}_{args.attack}{args.suffix}.pt'
    model = torch.load(model_filepath, map_location='cpu')
    model = model.cuda()
    model.eval()

    preprocess, _ = get_norm(args.dataset)

    num_classes = get_config(args.dataset)['num_classes']

    # Finetune dataset
    train_set = get_dataset(args, train=True, augment=True)
    train_set = FinetuneDataset(train_set, num_classes=num_classes, data_rate=args.ratio)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)

    # Test dataset
    test_set = get_dataset(args, train=False)

    if args.attack == 'clean':
        poison_set = test_set
    else:
        poison_set = get_poison_testset(args, test_set)

    poison_loader = DataLoader(dataset=poison_set, batch_size=args.batch_size)
    test_loader   = DataLoader(dataset=test_set, batch_size=args.batch_size)

    if Print_level > 0:
        print(f'Finetune dataset: {len(train_set)}, Test dataset: {len(test_set)}, Poison dataset: {len(poison_set)}')

    if preeval:
        acc, asr = test(model, test_loader, poison_loader, preprocess)
        print(f'Pre-evaluation of samples -> ACC: {acc*100:.2f}%, ASR: {asr*100:.2f}%')

    model, total_time = seam(args, model, train_loader, test_loader, poison_loader, preprocess)

    # Evaluate on the result model
    acc, asr = test(model, test_loader, poison_loader, preprocess)
    print(f'SEAM :: Dataset: {args.dataset}, Network: {args.network}, Attack: {args.attack} --- ACC: {acc*100:.2f}%, ASR: {asr*100:.2f}%, Time: {total_time:.2f}s')
    return acc, asr


if __name__ == '__main__':
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

    # Print arguments
    print_args(args)

    # Set seed
    seed_torch(args.seed)

    # Conduct experiment
    main(args)
