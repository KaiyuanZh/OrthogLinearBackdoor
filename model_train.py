# coding: utf-8

import warnings
from xml.dom import xmlbuilder
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import os
import sys
import time
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils.attack import Attack
from utils.utils import *


def eval_acc(model, loader, preprocess):
    model.eval()
    n_sample = 0
    n_correct = 0
    with torch.no_grad():
        for _, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            output = model(preprocess(x_batch))
            pred = output.max(dim=1)[1]

            n_sample  += x_batch.size(0)
            n_correct += (pred == y_batch).sum().item()

    acc = n_correct / n_sample
    return acc


def train(args):
    model = get_model(args).to(DEVICE)

    trainset = get_dataset(args, train=True)
    testset  = get_dataset(args, train=False)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(testset,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    preprocess, _ = get_norm(args.dataset)

    criterion = torch.nn.CrossEntropyLoss()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    else:
        raise ValueError('Optimizer not supported.')

    time_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            output = model(preprocess(x_batch))
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            pred = output.max(dim=1)[1]
            acc = (pred == y_batch).sum().item() / x_batch.size(0)

            if step % 10 == 0:
                sys.stdout.write('\repoch {:3}, step: {:4}, loss: {:.4f}, '
                                 .format(epoch, step, loss) +\
                                 'acc: {:.4f}'.format(acc))
                sys.stdout.flush()

        time_end = time.time()
        acc = eval_acc(model, test_loader, preprocess)

        sys.stdout.write('\repoch {:3}, step: {:4} - {:5.2f}s, '
                         .format(epoch, step, time_end-time_start) +\
                         'loss: {:.4f}, acc: {:.4f}\n'.format(loss, acc))
        sys.stdout.flush()
        time_start = time.time()

        scheduler.step()

        save_path = f'ckpt/{args.dataset}_{args.network}_clean'
        if args.suffix != '':
            save_path += f'_{args.suffix}.pt'
        else:
            save_path += '.pt'
        torch.save(model, save_path)


def test(args):
    model_filepath = f'ckpt/{args.dataset}_{args.network}_{args.attack}'
    if args.suffix != '':
        model_filepath += f'_{args.suffix}.pt'
    else:
        model_filepath += '.pt'

    model = torch.load(model_filepath, map_location='cpu').to(DEVICE)
    model.eval()

    preprocess, _ = get_norm(args.dataset)

    test_set = get_dataset(args, train=False)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=args.batch_size)

    acc = eval_acc(model, test_loader, preprocess)

    if args.attack == 'clean':
        print(f'Accuarcy: {acc*100:.2f}%')
    
    else:
        if args.attack == 'composite':
            mixer = HalfMixer()
            ca, cb, cc = 0, 1, 2
            poison_set = MixDataset(dataset=test_set, mixer=mixer, classA=ca,
                                    classB=cb, classC=cc, data_rate=1,
                                    normal_rate=0, mix_rate=0,
                                    poison_rate=1)
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
        
        poison_loader = DataLoader(dataset=poison_set, num_workers=0, batch_size=args.batch_size, shuffle=False)

        asr = eval_acc(model, poison_loader, preprocess)
        print(f'Accuarcy: {acc*100:.2f}%, ASR: {asr*100:.2f}%')


def train_mask(attack, train_loader):
    print('-'*70)
    print('Training mask...')
    print('-'*70)
    attack.backdoor.net_mask.train()

    for epoch in range(25):
        for step, (x_batch, _) in enumerate(train_loader):
            x_batch = x_batch.to(DEVICE)

            size = x_batch.size(0) // 2
            xb1 = x_batch[:size]
            xb2 = x_batch[size:]

            attack.optim_mask.zero_grad()
            masks1 = attack.backdoor.threshold(attack.backdoor.net_mask(xb1))
            masks2 = attack.backdoor.threshold(attack.backdoor.net_mask(xb2))

            div_input = attack.criterion_div(xb1, xb2)
            div_input = torch.mean(div_input, dim=(1, 2, 3))
            div_input = torch.sqrt(div_input)

            div_mask = attack.criterion_div(masks1, masks2)
            div_mask = torch.mean(div_mask, dim=(1, 2, 3))
            div_mask = torch.sqrt(div_mask)

            loss_norm = torch.mean(F.relu(masks1 - attack.mask_density))
            loss_div  = torch.mean(div_input / (div_mask + EPSILON))

            loss = attack.lambda_norm * loss_norm + attack.lambda_div * loss_div
            loss.backward()
            attack.optim_mask.step()

            if step % 10 == 0:
                sys.stdout.write('\repoch {:3}, step: {:4}, loss: {:.4f}, '
                                 .format(epoch, step, loss) +\
                                 f'norm: {loss_norm:.4f}, div: {loss_div:.4f}')
                sys.stdout.flush()

        attack.sched_mask.step()
        print()

    attack.backdoor.net_mask.eval()
    attack.backdoor.net_mask.requires_grad_(False)
    print('-'*70)


def poison(args):
    model = get_model(args).to(DEVICE)
    if args.dataset == 'gtsrb' and args.network == 'prn34':
        model = torch.load(f'ckpt/{args.dataset}_{args.network}_clean.pt', map_location='cpu').to(DEVICE)

    attack = Attack(model, args, device=DEVICE)

    workers = 0

    train_loader  = DataLoader(dataset=attack.train_set,  num_workers=workers,
                               batch_size=args.batch_size, shuffle=True)
    poison_loader = DataLoader(dataset=attack.poison_set, num_workers=0,
                               batch_size=args.batch_size)
    test_loader   = DataLoader(dataset=attack.test_set,   num_workers=4,
                               batch_size=args.batch_size)

    preprocess, _ = get_norm(args.dataset)

    save_path = f'ckpt/{args.dataset}_{args.network}_{args.attack}'
    if args.suffix != '':
        save_path += f'_{args.suffix}.pt'
    else:
        save_path += '.pt'
    
    trigger_path = f'data/trigger/{args.attack}/{args.dataset}_{args.network}'
    if args.suffix != '':
        trigger_path += f'_{args.suffix}'

    if args.attack == 'inputaware':
        train_mask(attack, train_loader)
        torch.save(attack.backdoor.net_mask, f'{trigger_path}_mask.pt')

    best_acc = 0
    best_asr = 0
    time_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        if args.attack in ['inputaware', 'dynamic']:
            attack.backdoor.net_genr.train()
        if args.attack == 'lira' and epoch < args.epochs // 2:
            attack.backdoor.net_genr.train()

        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            x_batch, y_batch = attack.inject(x_batch, y_batch)

            attack.optimizer.zero_grad()
            if args.attack in ['inputaware', 'dynamic']:
                attack.optim_genr.zero_grad()
            if args.attack == 'lira' and epoch < args.epochs // 2:
                attack.optim_genr.zero_grad()
            
            output = model(preprocess(x_batch))
            loss = attack.criterion(output, y_batch)
            if args.attack == 'inputaware':
                loss = loss + attack.loss_div

            loss.backward()
            attack.optimizer.step()
            if args.attack in ['inputaware', 'dynamic']:
                attack.optim_genr.step()
            if args.attack == 'lira' and epoch < args.epochs // 2:
                attack.optim_genr.step()

            pred = output.max(dim=1)[1]
            acc = (pred == y_batch).sum().item() / x_batch.size(0)

            if step % 10 == 0:
                sys.stdout.write('\repoch {:3}, step: {:4}, loss: {:.4f}, '
                                 .format(epoch, step, loss) +\
                                 'acc: {:.4f}'.format(acc))
                sys.stdout.flush()

        attack.scheduler.step()
        if args.attack in ['inputaware', 'dynamic']:
            attack.sched_genr.step()
            attack.backdoor.net_genr.eval()
        if args.attack == 'lira' and epoch < args.epochs // 2:
            attack.sched_genr.step()
            attack.backdoor.net_genr.eval()

        time_end = time.time()
        acc = eval_acc(model, test_loader, preprocess)
        asr = eval_acc(model, poison_loader, preprocess)

        sys.stdout.write('\repoch {:3}, step: {:4} - {:5.2f}s, acc: {:.4f}, '
                         .format(epoch, step, time_end-time_start, acc) +\
                         'asr: {:.4f}\n'.format(asr))
        sys.stdout.flush()
        time_start = time.time()

        if epoch > 10 and acc + asr > best_acc + best_asr:
            best_acc = acc
            best_asr = asr
            print(f'---BEST ACC: {best_acc:.4f}, ASR: {best_asr:.4f}---')
            torch.save(model, save_path)
            if args.attack in ['inputaware', 'dynamic', 'lira']:
                torch.save(attack.backdoor.net_genr, f'{trigger_path}_genr.pt')
        
        # Save intermediate results
        if (epoch + 1) % 10 == 0:
            torch.save(model, f'{save_path[:-3]}_epoch_{epoch+1}.pt')
            if args.attack in ['inputaware', 'dynamic', 'lira']:
                torch.save(attack.backdoor.net_genr, f'{trigger_path}_genr_epoch_{epoch+1}.pt')


def save_poisoned_data(args):
    model_filepath = f'ckpt/{args.dataset}_{args.network}_{args.attack}.pt'
    model = torch.load(model_filepath, map_location='cpu')

    model = model.to(DEVICE)
    model.eval()

    # Save raw poisoned data
    assert args.attack in ['invisible', 'dfst']

    shape = get_config(args.dataset)['size']
    backdoor = get_backdoor(args.attack, shape, DEVICE)

    testset = get_dataset(args, train=False)
    test_loader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False)

    savepath = f'data/{args.dataset}_{args.attack}.pt'

    x_poison, y_poison = [], []
    for _, (images, labels) in enumerate(test_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Poison
        images = backdoor.inject(images)

        x_poison.append(images.cpu())
        y_poison.append(labels.cpu())
    
    x_poison = torch.cat(x_poison, dim=0)
    y_poison = torch.cat(y_poison, dim=0)
    poison_set = torch.utils.data.TensorDataset(x_poison, y_poison)
    # save_image(x_poison[:100], 'tmp.png')
    torch.save(poison_set, savepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')

    parser.add_argument('--datadir',     default='./data',    help='root directory of data')
    parser.add_argument('--suffix',      default='',          help='suffix of saved path')
    parser.add_argument('--gpu',         default='0',         help='gpu id')

    parser.add_argument('--phase',       default='test',      help='phase of framework')
    parser.add_argument('--dataset',     default='cifar10',   help='dataset')
    parser.add_argument('--network',     default='resnet18',  help='network structure')
    parser.add_argument('--attack',      default='clean',     help='attack type')
    parser.add_argument('--optimizer',   default='sgd',       help='optimizer')

    parser.add_argument('--seed',        type=int, default=1024, help='seed index')
    parser.add_argument('--batch_size',  type=int, default=128,  help='attack size')
    parser.add_argument('--epochs',      type=int, default=100,  help='number of epochs')
    parser.add_argument('--target',      type=int, default=0,    help='target label')

    parser.add_argument('--poison_rate', type=float, default=0.1,  help='poisoning rate')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    DEVICE = torch.device('cuda')

    if args.phase == 'train':
        train(args)
    elif args.phase == 'test':
        test(args)
    elif args.phase == 'poison':
        poison(args)
    elif args.phase == 'save_poisoned_data':
        save_poisoned_data(args)
    else:
        raise NotImplementedError
