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

from attack import Attack
from factor_utils import *


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


def test(args):
    # Suffix
    if args.troj_type == 'univ-rate':
        suffix = f'{args.poison_rate}'
    elif args.troj_type == 'label-spec':
        suffix = f'{args.victim}-{args.target}'
    else:
        suffix = f'{args.troj_param}'
    
    #  best or final
    model_filepath = f'ckpt_{args.troj_type}/{args.dataset}_{args.network}_{args.attack}_{suffix}_final.pt'

    model = torch.load(model_filepath, map_location='cpu').to(DEVICE)
    model.eval()

    preprocess, _ = get_norm(args.dataset)

    test_set = get_dataset(args, train=False)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=args.batch_size)

    acc = eval_acc(model, test_loader, preprocess)

    #  evaluate ASR on noise data (args.troj_type in ['label-spec', 'trig-focus'])

    if args.attack == 'clean':
        print(f'Accuarcy: {acc*100:.2f}%')
    else:
        poison_set = PoisonDataset(dataset=test_set,
                                   attack=args.attack,
                                   victim=args.victim,
                                   target=args.target,
                                   poison_rate=1)

        poison_loader = DataLoader(dataset=poison_set, batch_size=args.batch_size, shuffle=False)

        asr = eval_acc(model, poison_loader, preprocess)
        print(f'Accuarcy: {acc*100:.2f}%, ASR: {asr*100:.2f}%')


def poison(args):
    if args.troj_type in ['acti-sep', 'weig-cal']:
        model = torch.load(f'../orthogonal/ckpt/{args.dataset}_{args.network}_clean.pt', map_location='cpu').to(DEVICE)
        clean_model = torch.load(f'../orthogonal/ckpt/{args.dataset}_{args.network}_clean.pt', map_location='cpu').to(DEVICE)
        clean_model.eval()
    else:
        model = get_model(args).to(DEVICE)

    attack = Attack(model, args, device=DEVICE)

    train_loader  = DataLoader(dataset=attack.train_set, batch_size=args.batch_size, shuffle=True)
    poison_loader = DataLoader(dataset=attack.poison_set, batch_size=args.batch_size)
    test_loader   = DataLoader(dataset=attack.test_set, batch_size=args.batch_size)

    preprocess, _ = get_norm(args.dataset)

    # Save root
    # save_root = f'ckpt_{args.troj_type}'
    save_root = 'clip_adap_badnet'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    # Suffix
    if args.troj_type == 'univ-rate':
        suffix = f'{args.poison_rate}'
    elif args.troj_type == 'label-spec':
        suffix = f'{args.victim}-{args.target}'
    else:
        suffix = f'{args.troj_param}'

    # Save best model
    best_save_path = f'{save_root}/{args.dataset}_{args.network}_{args.attack}_{suffix}_best.pt'

    # Save final model
    final_save_path = f'{save_root}/{args.dataset}_{args.network}_{args.attack}_{suffix}_final.pt'

    # Start training
    best_acc = 0
    best_asr = 0
    time_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            x_batch, y_batch = attack.inject(x_batch, y_batch)

            attack.optimizer.zero_grad()

            if args.troj_type == 'acti-sep':
                output, activation = model.custom_forward(preprocess(x_batch))
                ce_loss = attack.criterion(output, y_batch)

                # ============== ACTIVATION SIMILARITY LOSS ============== #
                with torch.no_grad():
                    ref_output, ref_activation = clean_model.custom_forward(preprocess(x_batch))

                # Match posteriors of clean model and poisoned model
                #  Only take the poisoned samples into account
                n_poison = int(x_batch.size(0) * args.poison_rate)
                # cl_act_layer3 = ref_activation['layer3'].detach()[:n_poison]
                # po_act_layer3 = activation['layer3'][:n_poison]
                # cl_act_layer4 = ref_activation['layer4'].detach()[:n_poison]
                # po_act_layer4 = activation['layer4'][:n_poison]
                # mse = nn.MSELoss()

                # acti_sim_loss = mse(po_act_layer3, cl_act_layer3) + mse(po_act_layer4, cl_act_layer4)
                cl_act_layer1 = ref_activation['layer1'].detach()[:n_poison]
                po_act_layer1 = activation['layer1'][:n_poison]
                cl_act_layer2 = ref_activation['layer2'].detach()[:n_poison]
                po_act_layer2 = activation['layer2'][:n_poison]
                cl_act_layer3 = ref_activation['layer3'].detach()[:n_poison]
                po_act_layer3 = activation['layer3'][:n_poison]
                cl_act_layer4 = ref_activation['layer4'].detach()[:n_poison]
                po_act_layer4 = activation['layer4'][:n_poison]
                mse = nn.MSELoss()

                acti_sim_loss = mse(po_act_layer1, cl_act_layer1) + mse(po_act_layer2, cl_act_layer2) + mse(po_act_layer3, cl_act_layer3) + mse(po_act_layer4, cl_act_layer4)

                # Default: 0.1
                reg_weight = float(args.troj_param)
                loss = ce_loss + reg_weight * acti_sim_loss

            elif args.troj_type == 'weig-cal':
                output = model(preprocess(x_batch))
                ce_loss = attack.criterion(output, y_batch)

                # ============== PARAMETER SIMILARITY LOSS ============== #
                param_sim_loss = 0
                for p1, p2 in zip(model.parameters(), clean_model.parameters()):
                    param_sim_loss += (p1 - p2.data.detach()).pow(2).sum()
                param_sim_loss = (param_sim_loss + 1e-12).pow(0.5)

                # Default: 0.05
                reg_weight = float(args.troj_param)
                loss = ce_loss + reg_weight * param_sim_loss

            else:
                # ============== DEFAULT ============== #
                output = model(preprocess(x_batch))
                loss = attack.criterion(output, y_batch)

            loss.backward()
            attack.optimizer.step()

            pred = output.max(dim=1)[1]
            if y_batch.dim() == 2:
                y_batch = y_batch.max(dim=1)[1]
            acc = (pred == y_batch).sum().item() / x_batch.size(0)

            if step % 10 == 0:
                sys.stdout.write('\repoch {:3}, step: {:4}, loss: {:.4f}, '
                                 .format(epoch, step, loss) +\
                                 'acc: {:.4f}'.format(acc))
                sys.stdout.flush()

        attack.scheduler.step()

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
            torch.save(model, best_save_path)

    torch.save(model, final_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')

    parser.add_argument('--datadir',     default='./data',    help='root directory of data')
    parser.add_argument('--suffix',      default='',          help='suffix of saved path')
    parser.add_argument('--gpu',         default='0',         help='gpu id')

    parser.add_argument('--phase',       default='test',      help='phase of framework')
    parser.add_argument('--dataset',     default='cifar10',   help='dataset')
    parser.add_argument('--network',     default='resnet18',  help='network structure')
    parser.add_argument('--attack',      default='clean',     help='attack type')

    parser.add_argument('--troj_type',   default='universal', help='trojan type')
    parser.add_argument('--troj_param',  default=None,        help='trojan parameter')

    parser.add_argument('--victim',      type=int, default=-1,   help='victim label')
    parser.add_argument('--target',      type=int, default=0,    help='target label')
    parser.add_argument('--poison_rate', type=float, default=0.1,  help='poisoning rate')

    parser.add_argument('--batch_size',  type=int, default=128,  help='attack size')
    parser.add_argument('--epochs',      type=int, default=100,  help='number of epochs')
    parser.add_argument('--seed',        type=int, default=1024, help='seed index')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    DEVICE = torch.device(f'cuda:{args.gpu}')

    # Assertion
    assert args.troj_type in ['univ-rate', 'low-conf', 'label-spec', 'trig-focus', 'acti-sep', 'weig-cal']

    if args.phase == 'test':
        test(args)
    elif args.phase == 'poison':
        poison(args)
    else:
        raise NotImplementedError
