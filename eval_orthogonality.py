# coding: utf-8

import warnings
from xml.dom import xmlbuilder
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import os
import sys
import time
import math
import pickle
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import shap
from scipy.stats import wasserstein_distance
from utils.utils import *
from backdoors import *


def eval_acc(model, loader, preprocess):
    model.eval()
    n_sample = 0
    n_correct = 0
    with torch.no_grad():
        for _, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            output = model(preprocess(x_batch))
            pred = output.max(dim=1)[1]

            n_sample += x_batch.size(0)
            n_correct += (pred == y_batch).sum().item()

    acc = n_correct / n_sample
    return acc


def get_poison_loader(args, testset, get_len=False):
    # Saved poisoned dataset
    poison_path = f'data/{args.dataset}_{args.attack}.pt'
    
    if args.attack == 'composite':
        # A + B -> C
        CLASS_A = 0
        CLASS_B = 1
        CLASS_C = 2

        mixer = HalfMixer()
        poison_set = MixDataset(dataset=testset, mixer=mixer,
                               classA=CLASS_A, classB=CLASS_B,
                               classC=CLASS_C, data_rate=1,
                               normal_rate=0, mix_rate=0,
                               poison_rate=1)
        poison_loader = torch.utils.data.DataLoader(poison_set, batch_size=args.batch_size, shuffle=False)
        # Re-label
        args.target = CLASS_C
    
    elif args.attack in ['invisible', 'dfst'] and os.path.exists(poison_path):
        # print(f'Loading saved poisoned ({args.attack}) dataset...')
        poison_set = torch.load(poison_path)
        x_poison, y_poison = poison_set.tensors
        new_x_poison, new_y_poison = [], []
        for i in range(len(y_poison)):
            if y_poison[i] != args.target:
                new_x_poison.append(x_poison[i])
                new_y_poison.append(y_poison[i] * 0 + args.target)
        x_poison = torch.stack(new_x_poison)
        y_poison = torch.stack(new_y_poison)
        poison_set = torch.utils.data.TensorDataset(x_poison, y_poison)
        poison_loader = torch.utils.data.DataLoader(poison_set, batch_size=args.batch_size, shuffle=False)
    
    else:
        shape = get_config(args.dataset)['size']
        backdoor = get_backdoor(args.attack, shape=shape, device=DEVICE)
        trigger_filepath = f'data/trigger/{args.attack}/{args.dataset}_{args.network}'
        # suffix = '_epoch_10'
        if args.attack == 'inputaware':
            backdoor.net_mask = torch.load(trigger_filepath + '_mask.pt', map_location='cpu').to(DEVICE)
            backdoor.net_mask.eval()
            backdoor.net_genr = torch.load(trigger_filepath + '_genr' + args.suffix + '.pt', map_location='cpu').to(DEVICE)
            backdoor.net_genr.eval()
        elif args.attack in ['dynamic', 'lira']:
            backdoor.net_genr = torch.load(trigger_filepath + '_genr' + args.suffix + '.pt', map_location='cpu').to(DEVICE)
            backdoor.net_genr.eval()

        poison_set = PoisonDataset(dataset=testset, threat='dirty', attack=args.attack, target=args.target, poison_rate=1, backdoor=backdoor)
        poison_loader = torch.utils.data.DataLoader(poison_set, batch_size=args.batch_size, shuffle=False)
    
    if get_len:
        return poison_loader, len(poison_set)
    else:
        return poison_loader


############################################################################
# Customized functions
############################################################################
class custom_relu(nn.Module):
    def __init__(self, acti_choice=None):
        super(custom_relu, self).__init__()
        self.acti_choice = acti_choice
    
    def forward(self, x, index):
        if self.acti_choice is None:
            return F.relu(x)
        elif self.acti_choice == 'linear':
            return x
        elif self.acti_choice == 'leaky_relu':
            return F.leaky_relu(x, negative_slope=0.1)
        else:
            # self.acti_choice is a mask for pruning
            return x * self.acti_choice[index]


class ResNet18_relu(nn.Module):
    def __init__(self, model, acti_choice=None):
        super(ResNet18_relu, self).__init__()
        self.model = model
        self.activation = custom_relu(acti_choice)
    
    def forward(self, x, save=False):
        if save:
            x_acti = {}
        
        # Pre-extract features
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        index = 'pre_extract'
        if save:
            x = F.relu(x)
            x_acti[index] = x
        else:
            x = self.activation(x, index)
        
        # Extract features
        for i in range(1, 5):
            block = getattr(self.model, 'layer{}'.format(i))
            for j in range(2):
                inputs = x
                x = block[j].conv1(x)
                x = block[j].bn1(x)
                index = 'layer{}_{}_0'.format(i, j)
                if save:
                    x = F.relu(x)
                    x_acti[index] = x
                else:
                    x = self.activation(x, index)
                x = block[j].conv2(x)
                x = block[j].bn2(x)
                x += block[j].shortcut(inputs)
                index = 'layer{}_{}_1'.format(i, j)
                if save:
                    x = F.relu(x)
                    x_acti[index] = x
                else:
                    x = self.activation(x, index)

        # Post-extract features
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.model.linear(x)

        if save:
            return x, x_acti
        else:
            return x


class ResNet18_last(nn.Module):
    def __init__(self, model):
        super(ResNet18_last, self).__init__()
        self.model = model
    
    def forward(self, x):
        # Post-extract features
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.model.linear(x)

        return x


class wrn_relu(nn.Module):
    def __init__(self, model, acti_choice=None):
        super(wrn_relu, self).__init__()
        self.model = model
        self.activation = custom_relu(acti_choice)
    
    def forward(self, x, save=False):
        if save:
            x_acti = {}
        
        # Pre-extract features
        x = self.model.conv1(x)

        # Extract features
        for i in range(1, 4):
            block = getattr(self.model, 'block{}'.format(i))
            for j in range(4):
                shortcut = x
                x = block.layer[j].bn1(x)

                index = f'block{i}_layer{j}'
                if save:
                    x = F.relu(x)
                    x_acti[f'{index}_1'] = x
                else:
                    x = self.activation(x, f'{index}_1')
                if block.layer[j].convShortcut is not None:
                    shortcut = block.layer[j].convShortcut(x)
                x = block.layer[j].conv1(x)
                x = block.layer[j].bn2(x)
                if save:
                    x = F.relu(x)
                    x_acti[f'{index}_2'] = x
                else:
                    x = self.activation(x, f'{index}_2')
                if block.layer[j].droprate > 0:
                    x = F.dropout(x, p=block.layer[j].droprate, training=block.layer[j].training)
                x = block.layer[j].conv2(x)
                x = x + shortcut
        
        # Post-extract features
        x = self.model.bn1(x)
        index = 'post-extract'
        if save:
            x = F.relu(x)
            x_acti[index] = x
        else:
            x = self.activation(x, index)
        
        out = F.avg_pool2d(x, 8)
        out = out.view(out.size(0), -1)
        out = self.model.fc(out)

        if save:
            return out, x_acti
        else:
            return out


class wrn_last(nn.Module):
    def __init__(self, model):
        super(wrn_last, self).__init__()
        self.model = model
    
    def forward(self, x):
        # Post-extract features
        x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)

        return x

############################################################################


def eval_performance(args):
    model_filepath = f'ckpt/{args.dataset}_{args.network}_{args.attack}{args.suffix}.pt'
    model = torch.load(model_filepath, map_location='cpu')

    model = model.to(DEVICE)
    model.eval()

    preprocess, _ = get_norm(args.dataset)

    test_set = get_dataset(args, train=False)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=args.batch_size)

    acc = eval_acc(model, test_loader, preprocess)

    if args.attack == 'clean':
        print(f'Accuracy ({len(test_set)}): {acc*100.:.2f}%')
    else:
        poison_loader, poison_len = get_poison_loader(args, test_set, get_len=True)
        asr = eval_acc(model, poison_loader, preprocess)
        print(f'Accuracy ({len(test_set)}): {acc*100.:.2f}%, ASR ({poison_len}): {asr*100.:.2f}%')




def eval_shap(args):
    model_filepath = f'ckpt/{args.dataset}_{args.network}_{args.attack}{args.suffix}.pt'
    model = torch.load(model_filepath, map_location='cpu')

    # Rewrite the model
    if args.network == 'resnet18':
        last_layer = ResNet18_last(model)
        model = ResNet18_relu(model)
        target_layer_id = 'layer4_1_1'
    elif args.network == 'wrn':
        last_layer = wrn_last(model)
        model = wrn_relu(model)
        # 'post-extract', 'block3_layer3_2'
        target_layer_id = 'post-extract'
    else:
        raise NotImplementedError

    model = model.to(DEVICE)
    model.eval()

    last_layer.eval()

    preprocess, _ = get_norm(args.dataset)

    testset = get_dataset(args, train=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    if args.attack == 'clean':
        poison_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)
    else:
        poison_loader = get_poison_loader(args, testset)

    clean_acti = []
    poison_acti = []

    # Step 1: Collect activation for clean and poisoned images (only 8 batches)
    n_batch = 1
    with torch.no_grad():
        for i in range(n_batch):
            x_clean, y_clean = next(iter(test_loader))
            x_poison, y_poison = next(iter(poison_loader))

            x_clean, y_clean = x_clean.to(DEVICE), y_clean.to(DEVICE)
            x_poison, y_poison = x_poison.to(DEVICE), y_poison.to(DEVICE)

            _, batch_acti_clean = model(preprocess(x_clean), save=True)
            _, batch_acti_poison = model(preprocess(x_poison), save=True)

            clean_acti.append(batch_acti_clean[target_layer_id].detach().cpu())
            poison_acti.append(batch_acti_poison[target_layer_id].detach().cpu())

    clean_acti = torch.cat(clean_acti, dim=0).to(DEVICE)
    poison_acti = torch.cat(poison_acti, dim=0).to(DEVICE)

    # Step 2: Select 3% neurons with highest sensitivity to poisoned images
    data = poison_acti
    # data: (n_batch, c, h, w)
    background = poison_acti # clean_acti
    explainer = shap.DeepExplainer(last_layer, background)

    # Calculate the SHAP values for the test set
    # Shap value: (n_classes, n_batch, c, h, w)
    # explainer = shap.GradientExplainer(last_layer, background)
    # tmp = explainer.shap_values(data)
    # print(data.shape)
    # print(len(tmp))
    # print(tmp[0].shape)
    # exit()
    shap_values = explainer.shap_values(data)[args.target].reshape(data.shape[0], data.shape[1], -1)
    shap_values = np.max(shap_values, axis=2)
    shap_values = shap_values.mean(axis=0)

    # TODO: Select the top-k% neurons
    if args.dataset == 'cifar10':
        _k = 0.01
    else:
        _k = 0.03
    n_select = int(shap_values.size * _k)
    selected_neurons = np.argsort(shap_values)[-n_select:]
    # print(f'Selected neurons: {selected_neurons}')

    # Take the activation for clean and poisoned for selected neurons
    clean_acti = clean_acti.view(clean_acti.size(0), clean_acti.size(1), -1).max(dim=2)[0][:, selected_neurons].detach().cpu().numpy()
    poison_acti = poison_acti.view(poison_acti.size(0), poison_acti.size(1), -1).max(dim=2)[0][:, selected_neurons].detach().cpu().numpy()

    # Step 3: Measure the linear separability
    ls = []
    for i in range(clean_acti.shape[1]):
        cl_act = clean_acti[:, i]
        po_act = poison_acti[:, i]
        was_dist = wasserstein_distance(cl_act, po_act)
        thickness = np.std(cl_act) + np.std(po_act)
        ls.append(was_dist / thickness)
    
    print(f'Linear separability: {np.mean(ls):.4f} +- {np.std(ls):.4f}')


def eval_orthogonal(args):
    model_filepath = f'ckpt/{args.dataset}_{args.network}_{args.attack}{args.suffix}.pt'
    model = torch.load(model_filepath, map_location='cpu')

    model = model.to(DEVICE)
    model.eval()

    testset = get_dataset(args, train=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    if args.attack == 'clean':
        target_set = TargetDataset(dataset=testset, target=args.target)
        poison_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size, shuffle=True)
    else:
        poison_loader = get_poison_loader(args, testset)

    # Calculate the gradient for clean and poisoned images
    clean_gradients = []
    poison_gradients = []

    n_batch = 8
    for i in range(n_batch):
        x_clean, y_clean = next(iter(test_loader))
        x_poison, y_poison = next(iter(poison_loader))

        x_clean, y_clean = x_clean.to(DEVICE), y_clean.to(DEVICE)
        x_poison, y_poison = x_poison.to(DEVICE), y_poison.to(DEVICE)

        batch_clean_grad = compute_all_layer_gradients(args, model, x_clean, y_clean)
        batch_poison_grad = compute_all_layer_gradients(args, model, x_poison, y_poison)

        # print(f"batch_clean_grad shape: {batch_clean_grad.shape}, batch_poison_grad shape: {batch_poison_grad.shape}")

        clean_gradients.append(batch_clean_grad)
        poison_gradients.append(batch_poison_grad)
    
    clean_gradients = torch.mean(torch.stack(clean_gradients), dim=0)
    poison_gradients = torch.mean(torch.stack(poison_gradients), dim=0)

    # print(f"clean_grad shape: {clean_gradients.shape}, poison_grad shape: {poison_gradients.shape}")

    cosine_similarity = torch.nn.functional.cosine_similarity(clean_gradients, poison_gradients, dim=0)
    angle = torch.acos(cosine_similarity) * 180 / math.pi
    print("=====================================")
    print(f"cosine_similarity {cosine_similarity.item()}, angle {angle.item()}")


def compute_all_layer_gradients(args, model, inputs, labels):
    preprocess, _ = get_norm(args.dataset)

    model.zero_grad()
    output = model(preprocess(inputs))

    # if args.attack == 'composite':
    #     CLASS_A = 0
    #     CLASS_B = 1
    #     CLASS_C = 2  # A + B -> C
    #     criterion = CompositeLoss(rules=[(CLASS_A,CLASS_B,CLASS_C)], simi_factor=1, mode='contrastive', device=DEVICE)
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()

    criterion = torch.nn.CrossEntropyLoss()

    loss = criterion(output, labels)
    loss.backward()

    gradients = []

    for name, p in model.named_parameters():
        if 'conv' in name:
            grad = p.grad.clone().abs().detach()
            # print(f"gradients shape: {grad.shape}")
            gradients.append(grad.cpu().view(-1))
    gradients = torch.cat(gradients)
    return gradients


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--datadir', default='./data', help='root directory of data')

    parser.add_argument('--phase', default='orthogonal', help='phase')
    parser.add_argument('--dataset', default='cifar10', help='dataset')
    parser.add_argument('--network', default='resnet18', help='network structure')
    parser.add_argument('--suffix', default='', help='suffix of the model')

    parser.add_argument('--attack', default='badnet', help='attack method')
    parser.add_argument('--target', type=int, default=0, help='target class')
    parser.add_argument('--poison_rate', type=float, default=0.1,  help='poisoning rate')

    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--seed', type=int, default=123, help='seed index')

    args = parser.parse_args()

    # Print arguments
    # print_args(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    DEVICE = torch.device(f'cuda:{args.gpu}')

    if args.phase == 'test':
        eval_performance(args)
    elif args.phase == 'orthogonal':
        eval_orthogonal(args)
