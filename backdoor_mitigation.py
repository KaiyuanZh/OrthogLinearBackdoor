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


##############################################################################
# Baseline 1: Fine-tuning
##############################################################################
def finetune(args, model, train_loader, test_loader, poison_test_loader, preprocess):
    # Set hyperparameters for lr=1e-1
    lr = 2e-2

    epochs = 10
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    total_time = 0
    time_start = time.time()
    for epoch in range(epochs):
        model.train()
        if epoch > 0 and epoch % 2 == 0:
            lr /= 10

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for step, (x_batch, y_batch) in enumerate(train_loader):
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
# Baseline 2: Fine-prune
##############################################################################
def fineprune(args, model, train_loader, test_loader, poison_test_loader, preprocess):
    if args.network == 'wrn':
        net_pruned = copy.deepcopy(model)
    else:
        # pruning
        container = []

        def forward_hook(module, input, output):
            container.append(output)

        if args.network == 'vgg11':
            hook = model.features[25].register_forward_hook(forward_hook)
        elif args.network == 'resnet18':
            hook = model.layer4.register_forward_hook(forward_hook)
        elif args.network == 'wrn':
            hook = model.block3.register_forward_hook(forward_hook)

        for i, (x_test, _) in enumerate(test_loader):
            x_test = x_test.cuda()
            model(preprocess(x_test))
            if i * args.batch_size >= 1000:
                break

        container = torch.cat(container, dim=0)
        activation = torch.mean(container, dim=[0, 2, 3])
        seq_sort = torch.argsort(activation)
        pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)
        hook.remove()

        net_pruned = copy.deepcopy(model)
        num_pruned = int(0.2 * len(seq_sort))
        seq_sort = seq_sort[:num_pruned]
        pruning_mask[seq_sort] = False

        num_classes = get_config(args.dataset)['num_classes']

        if args.network == 'vgg11':
            net_pruned.features[25] = nn.Conv2d(
                pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            )
            net_pruned.features[26] = nn.BatchNorm2d(pruning_mask.shape[0] - num_pruned)
            net_pruned.classifier = nn.Linear(pruning_mask.shape[0] - num_pruned, num_classes)
        elif args.network == 'resnet18':
            net_pruned.layer4[1].conv2 = nn.Conv2d(
                pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            )
            net_pruned.layer4[1].bn2 = nn.BatchNorm2d(pruning_mask.shape[0] - num_pruned)
            net_pruned.layer4[1].shortcut = nn.Sequential(
                nn.Conv2d(pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(pruning_mask.shape[0] - num_pruned),
            )
        elif args.network == 'wrn':
            net_pruned.block3[1].conv2 = nn.Conv2d(
                pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            )
            net_pruned.block3[1].bn2 = nn.BatchNorm2d(pruning_mask.shape[0] - num_pruned)
            net_pruned.block3[1].convShortcut = nn.Sequential(
                nn.Conv2d(pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, kernel_size=1, stride=1, bias=False),
            )

        for name, module in net_pruned._modules.items():
            if 'layer4' in name:
                module[1].conv2.weight.data = model.layer4[1].conv2.weight.data[pruning_mask]
                module[1].ind = pruning_mask
            elif 'linear' == name:
                module.weight.data = model.linear.weight.data[:, pruning_mask]
                module.bias.data = model.linear.bias.data
            elif 'block3' in name:
                module[1].conv2.weight.data = model.block3[1].conv2.weight.data[pruning_mask]
                module[1].ind = pruning_mask
            elif 'fc' == name:
                module.weight.data = model.fc.weight.data[:, pruning_mask]
                module.bias.data = model.fc.bias.data
            else:
                continue
    
    net_pruned.cuda()
    lr = 1e-2

    if args.dataset == 'gtsrb':
        lr = 2e-2

    epochs = 10
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net_pruned.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    total_time = 0
    time_start = time.time()
    for epoch in range(epochs):
        net_pruned.train()

        # if epoch > 0 and epoch % 2 == 0:
        #     lr /= 10

        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr

        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

            optimizer.zero_grad()

            output = net_pruned(preprocess(x_batch))
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 2 == 0:
            time_end = time.time()

            net_pruned.eval()
            correct_cl = 0
            correct_bd = 0

            with torch.no_grad():
                total_cl = 0
                for (x_test, y_test) in test_loader:
                    x_test = x_test.cuda()
                    y_test = y_test.cuda()
                    total_cl += y_test.size(0)

                    ### clean accuracy ###
                    y_out = net_pruned(preprocess(x_test))
                    _, y_pred = torch.max(y_out.data, 1)
                    correct_cl += (y_pred == y_test).sum().item()
                
                total_bd = 0
                for (x_test, y_test) in poison_test_loader:
                    x_test = x_test.cuda()
                    y_test = y_test.cuda()
                    total_bd += y_test.size(0)

                    ### backdoor accuracy ###
                    y_out = net_pruned(preprocess(x_test))
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

    return net_pruned, total_time


##############################################################################
# Baseline 3: NAD
##############################################################################
class AT(nn.Module):
    def __init__(self, p):
        super(AT, self).__init__()
        self.p = p

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))
        return loss

    def attention_map(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2,3), keepdim=True)
        am = torch.div(am, norm+eps)
        return am


def adjust_learning_rate(optimizer, epoch):
    if epoch < 2:
        lr = 1e-3
    elif epoch < 4:
        lr = 1e-3
    else:
        lr = 1e-4
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def nad(args, model, train_loader, test_loader, poison_test_loader, preprocess):
    teacher, _ = finetune(args, model, train_loader, test_loader, poison_test_loader, preprocess)
    student = copy.deepcopy(teacher)

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    lr = 1e-2
    epochs = 6
    beta3 = 5000

    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_at = AT(2.0)
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    total_time = 0
    time_start = time.time()
    for epoch in range(epochs):
        student.train()
        lr = adjust_learning_rate(optimizer, epoch)

        activation = {}
        def get_activation(name):
            def hook(student, input, output):
                activation[name] = output
            return hook
        
        if args.network == 'vgg11':
            shandle = student.features.register_forward_hook(get_activation('snet_feature'))
            thandle = teacher.features.register_forward_hook(get_activation('tnet_feature'))
        if args.network == 'resnet18':
            shandle = student.layer4.register_forward_hook(get_activation('snet_feature'))
            thandle = teacher.layer4.register_forward_hook(get_activation('tnet_feature'))
        if args.network == 'wrn':
            shandle = student.block3.register_forward_hook(get_activation('snet_feature'))
            thandle = teacher.block3.register_forward_hook(get_activation('tnet_feature'))

        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

            output_s = student(preprocess(x_batch))
            activation3_s = activation['snet_feature']

            with torch.no_grad():
                _ = teacher(preprocess(x_batch))
                activation3_t = activation['tnet_feature']

            ce_loss  = criterion_ce(output_s, y_batch)
            at3_loss = criterion_at(activation3_s, activation3_t)
            loss = ce_loss + beta3 * at3_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Remove hooks
        shandle.remove()
        thandle.remove()

        if (epoch+1) % 1 == 0:
            time_end = time.time()

            student.eval()

            correct_cl = 0
            correct_bd = 0

            with torch.no_grad():
                total_cl = 0
                for (x_test, y_test) in test_loader:
                    x_test = x_test.cuda()
                    y_test = y_test.cuda()
                    total_cl += y_test.size(0)

                    ### clean accuracy ###
                    y_out = student(preprocess(x_test))
                    _, y_pred = torch.max(y_out.data, 1)
                    correct_cl += (y_pred == y_test).sum().item()
                
                total_bd = 0
                for (x_test, y_test) in poison_test_loader:
                    x_test = x_test.cuda()
                    y_test = y_test.cuda()
                    total_bd += y_test.size(0)

                    ### backdoor accuracy ###
                    y_out = student(preprocess(x_test))
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

    return student, total_time


##############################################################################
# Baseline 4: ANP
##############################################################################
class Hook():
    def __init__(self, name, module):
        self.name = name
        self.weight = module.weight
        self.bias   = module.bias
        self.running_mean = module.running_mean
        self.running_var  = module.running_var
        self.eps = module.eps
        self.momentum = module.momentum
        self.num_features = self.weight.shape[0]

        self.neuron_mask = Parameter(torch.Tensor(self.num_features).cuda())
        self.neuron_noise = Parameter(torch.Tensor(self.num_features).cuda())
        self.neuron_noise_bias = Parameter(torch.Tensor(self.num_features).cuda())
        nn.init.ones_(self.neuron_mask)
        nn.init.zeros_(self.neuron_noise)
        nn.init.zeros_(self.neuron_noise_bias)

        self.is_perturbed = False
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        if self.is_perturbed:
            coeff_weight = self.neuron_mask + self.neuron_noise
            coeff_bias = 1.0 + self.neuron_noise_bias
        else:
            coeff_weight = self.neuron_mask
            coeff_bias = 1.0
        return F.batch_norm(input[0], self.running_mean, self.running_var,
                            self.weight * coeff_weight, self.bias * coeff_bias,
                            False, self.momentum, self.eps)

    def close(self):
        self.hook.remove()


def anp(args, model, train_loader, test_loader, poison_test_loader, preprocess):
    # Record original state
    state_dict = model.state_dict()

    def reset(noise_params, rand_init=False, eps=0.0):
        for neuron_noise in noise_params:
            if rand_init:
                nn.init.uniform_(neuron_noise, a=-eps, b=eps)
            else:
                nn.init.zeros_(neuron_noise)

    def include_noise(hooks):
        for hook in hooks:
            hook.is_perturbed = True

    def exclude_noise(hooks):
        for hook in hooks:
            hook.is_perturbed = False

    def sign_grad(noise_params):
        for p in noise_params:
            p.grad.data = torch.sign(p.grad.data)

    def clip_mask(mask_params, lower=0.0, upper=1.0):
        with torch.no_grad():
            for param in mask_params:
                param.clamp_(lower, upper)
    
    hooks = []
    mask_params  = []
    noise_params = []
    
    for name, module in model.named_modules():
        if args.network == 'vgg11':
            target_layers = ['features.1', 'features.5', 'features.9', 'features.12', 'features.16', 'features.19', 'features.23', 'features.26']
            if name in target_layers:
                hook = Hook(name, module)
                hooks.append(hook)
                mask_params.append(hook.neuron_mask)
                noise_params.append(hook.neuron_noise)
                noise_params.append(hook.neuron_noise_bias)
        elif args.network in ['resnet18', 'wrn']:
            if 'bn' in name:
                hook = Hook(name, module)
                hooks.append(hook)
                mask_params.append(hook.neuron_mask)
                noise_params.append(hook.neuron_noise)
                noise_params.append(hook.neuron_noise_bias)
    
    # Hyperparameters
    lr = 0.1
    epochs = 10

    anp_steps = 1
    # Small anp_eps and large anp_alpha lead to high accuracy
    # anp_eps = 0.2
    # anp_alpha = 0.9
    anp_eps = 0.4
    anp_alpha = 0.5

    if args.attack == 'wanet':
        anp_eps = 0.3
        anp_alpha = 0.4
    
    if args.dataset == 'gtsrb':
        # anp_eps = 0.27
        anp_alpha = 0.2
        anp_alpha = 0.9

    criterion = torch.nn.CrossEntropyLoss()
    mask_optimizer  = torch.optim.SGD(mask_params,  lr=lr, momentum=0.9)
    noise_optimizer = torch.optim.SGD(noise_params, lr=anp_eps / anp_steps)

    total_time = 0
    time_start = time.time()
    for epoch in range(epochs):
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

            # step 1: calculate the adversarial perturbation for neurons
            reset(noise_params, rand_init=True, eps=anp_eps)
            for _ in range(anp_steps):
                noise_optimizer.zero_grad()

                include_noise(hooks)
                output_noise = model(preprocess(x_batch))
                loss_noise = - criterion(output_noise, y_batch)

                loss_noise.backward()
                sign_grad(noise_params)
                noise_optimizer.step()

            # step 2: calculate loss and update the mask values
            mask_optimizer.zero_grad()
            include_noise(hooks)
            output_noise = model(preprocess(x_batch))
            loss_rob = criterion(output_noise, y_batch)

            exclude_noise(hooks)
            output_clean = model(preprocess(x_batch))
            loss_nat = criterion(output_clean, y_batch)
            loss = anp_alpha * loss_nat + (1 - anp_alpha) * loss_rob

            loss.backward()
            mask_optimizer.step()
            clip_mask(mask_params)

        if (epoch+1) % 1 == 0:
            time_end = time.time()

            total = 0
            correct_cl = 0
            with torch.no_grad():
                for (x_test, y_test) in test_loader:
                    x_test = x_test.cuda()
                    y_test = y_test.cuda()
                    total += y_test.size(0)

                    ### clean accuracy ###
                    y_out = model(preprocess(x_test))
                    _, y_pred = torch.max(y_out.data, 1)
                    correct_cl += (y_pred == y_test).sum().item()

            acc = correct_cl / total

            if Print_level > 0:
                sys.stdout.write('epoch: {:2}/{} - {:.2f}s, '
                                    .format(epoch+1, epochs, time_end-time_start)\
                                    + 'loss: {:.4f}, acc: {:.4f}\n'
                                    .format(loss, acc))
                sys.stdout.flush()

            total_time += (time_end-time_start)
            time_start = time.time()

            mask_values = {}
            for idx in range(len(mask_params)):
                mask_values[hooks[idx].name] = mask_params[idx].detach().cpu().numpy()

    # Pruning
    time_start = time.time()
    threshold = 0.2
    for key in mask_values.keys():
        mask = mask_values[key]
        for idx in range(len(mask)):
            if float(mask[idx]) <= threshold:
                weight_name = '{}.{}'.format(key, 'weight')
                state_dict[weight_name][idx] = 0.0
    model.load_state_dict(state_dict)
    total_time += (time.time()-time_start)

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

    if args.phase == 'finetune':
        model, total_time = finetune(args, model, train_loader, test_loader, poison_loader, preprocess)
    elif args.phase == 'nad':
        model, total_time = nad(args, model, train_loader, test_loader, poison_loader, preprocess)
    elif args.phase == 'anp':
        model, total_time = anp(args, model, train_loader, test_loader, poison_loader, preprocess)
    elif args.phase == 'fineprune':
        model, total_time = fineprune(args, model, train_loader, test_loader, poison_loader, preprocess)

    # Evaluate on the result model
    acc, asr = test(model, test_loader, poison_loader, preprocess)
    print(f'{args.phase} :: Dataset: {args.dataset}, Network: {args.network}, Attack: {args.attack} --- ACC: {acc*100:.2f}%, ASR: {asr*100:.2f}%, Time: {total_time:.2f}s')
    return acc, asr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--datadir', default='./data', help='root directory of data')

    parser.add_argument('--phase', default='finetune', help='detection method')
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
