import warnings
warnings.filterwarnings('ignore')

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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from utils.utils import *
from backdoors import *


def eval_acc(model, loader, preprocess):
    model.eval()
    n_sample = 0
    n_correct = 0
    with torch.no_grad():
        for _, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

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
        backdoor = get_backdoor(args.attack, shape=shape, device=torch.device('cuda'))
        trigger_filepath = f'data/trigger/{args.attack}/{args.dataset}_{args.network}'
        # suffix = '_epoch_10'
        if args.attack == 'inputaware':
            backdoor.net_mask = torch.load(trigger_filepath + '_mask.pt', map_location='cpu').cuda()
            backdoor.net_mask.eval()
            backdoor.net_genr = torch.load(trigger_filepath + '_genr' + args.suffix + '.pt', map_location='cpu').cuda()
            backdoor.net_genr.eval()
        elif args.attack in ['dynamic', 'lira']:
            backdoor.net_genr = torch.load(trigger_filepath + '_genr' + args.suffix + '.pt', map_location='cpu').cuda()
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
def sub_network(model, network):
    if network == 'resnet18':
        children = list(model.children())
        nchildren = []
        for c in children:
            if c.__class__.__name__ == 'SequentialWithArgs':
                nchildren += list(c.children())
            else:
                nchildren.append(c)
        children = nchildren
        children.insert(2, torch.nn.ReLU())
        children.insert(-1, torch.nn.AvgPool2d(4))
        children.insert(-1, torch.nn.Flatten())
        target_layers = ['BasicBlock', 'BatchNorm2d']
    elif network == 'wrn':
        children = list(model.children())
        nchildren = []
        for c in children:
            if c.__class__.__name__ == 'NetworkBlock':
                nchildren += list(c.layer.children())
            else:
                nchildren.append(c)
        children = nchildren
        children.insert(-1, torch.nn.AvgPool2d(8))
        children.insert(-1, torch.nn.Flatten())
        target_layers = ['BasicBlock', 'Conv2d']
    else:
        raise NotImplementedError

    # Find the target layers
    target_ids = []
    for i, c in enumerate(children):
        if c.__class__.__name__ in target_layers:
            target_ids.append(i)

    return children, target_ids


def split_model(children, target_id):
    model_head = torch.nn.Sequential(*children[:target_id])
    model_tail = torch.nn.Sequential(*children[target_id:])
    return model_head, model_tail


class Custom_model(nn.Module):
    def __init__(self, model):
        super(Custom_model, self).__init__()
        self.model = model

    def forward(self, x):
        for layer in self.model.children():
            if layer.__class__.__name__ == 'Flatten':
                x = x.view(x.size(0), -1)
            else:
                x = layer(x)
        return x


############################################################################


def eval_linear(args):
    model_filepath = f'ckpt/{args.dataset}_{args.network}_{args.attack}{args.suffix}.pt'
    model = torch.load(model_filepath, map_location='cpu')

    model = model.cuda()
    model.eval()

    preprocess, _ = get_norm(args.dataset)

    testset = get_dataset(args, train=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    if args.attack == 'clean':
        # TODO: Only take the images from the target class
        poison_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)
    else:
        poison_loader = get_poison_loader(args, testset)

    children, target_ids = sub_network(model, args.network)

    for target_id in target_ids:
        time_start = time.time()

        model_head, model_tail = split_model(children, target_id)

        model_head = Custom_model(model_head)
        model_tail = Custom_model(model_tail)

        # Get the output of the target layer
        with torch.no_grad():
            # Get data of clean images
            for _, (x_batch, y_batch) in enumerate(test_loader):
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                # Inputs for the tail model (Only 1 batch)
                background = model_head(preprocess(x_batch))
                break
            
            # Get the output of the tail model
            outputs = model_tail(background)
            pred = outputs.max(dim=1)[1]
            acc = (pred == y_batch).sum().item() / x_batch.size(0)

            # Get data of poisoned images
            for _, (x_batch, y_batch) in enumerate(poison_loader):
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                # Inputs for the tail model (Only 1 batch)
                data = model_head(preprocess(x_batch))
                break
            
            # Get the output of the tail model
            outputs = model_tail(data)
            pred = outputs.max(dim=1)[1]
            asr = (pred == y_batch).sum().item() / x_batch.size(0)

        print(f'Current target layer: {target_id}, Accuracy: {acc*100:.2f}%, ASR: {asr*100:.2f}%')

        # Use SHAP to identify the important neurons
        background = background[:16]
        explainer = shap.DeepExplainer(model_tail, background)

        # Calculate the SHAP values for the test set
        shap_values = explainer.shap_values(data)[args.target].reshape(data.shape[0], data.shape[1], -1)
        shap_values = np.max(shap_values, axis=2)
        shap_values = shap_values.mean(axis=0)

        # TODO: Select the top-k% neurons
        _k = 0.03
        n_select = int(np.ceil(shap_values.size * _k))
        selected_neurons = np.argsort(shap_values)[-n_select:]

        time_end = time.time()
        # print(f'Selected neurons: {selected_neurons}, time: {time_end - time_start}')

        n_neurons = len(selected_neurons)
        test_acti = data[:, selected_neurons].reshape(data.shape[0], n_neurons, -1).max(dim=2)[0]
        n_activated = ((test_acti > 1e-3).sum(dim=0) / test_acti.shape[0]) > 0.9
        n_activated = n_activated.sum().item()

        # Mutate the values of the selected neurons
        neuron_mask = torch.zeros_like(data)
        neuron_mask[:, selected_neurons] = 1

        layer_mean = data.mean(dim=[0], keepdim=True)

        linear_inputs = np.arange(0, 3, 0.1)
        # linear_inputs = np.arange(0, 1, 0.1)
        linear_outputs = []
        for w in linear_inputs:
            mute = w * layer_mean * neuron_mask
            data_mute = data + mute
            output = model_tail(data_mute)

            if w == 0:
                base = output
            else:
                diff = (output - base)[:, args.target].detach().cpu().numpy()
                linear_outputs.append(diff)

        linear_inputs = np.array(linear_inputs)[1:].reshape(-1, 1)
        linear_outputs = np.array(linear_outputs)

        # Measure the linearity of the mapping
        reg = LinearRegression().fit(linear_inputs, linear_outputs)
        r2 = r2_score(linear_outputs, reg.predict(linear_inputs))
        print(f'No. activated: {n_activated} ({n_neurons}), Linearity score: {r2}')
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--datadir', default='./data', help='root directory of data')

    parser.add_argument('--dataset', default='gtsrb', help='dataset')
    parser.add_argument('--network', default='wrn', help='network structure')
    parser.add_argument('--suffix', default='', help='suffix of the model')

    parser.add_argument('--attack', default='badnet', help='attack method')
    parser.add_argument('--target', type=int, default=0, help='target class')

    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--seed', type=int, default=123, help='seed index')

    args = parser.parse_args()

    # Print arguments
    # print_args(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    eval_linear(args)
