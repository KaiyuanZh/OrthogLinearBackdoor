import os
import sys
import time
import pickle
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image


def epsilon():
    return 1e-7


def mask_patn_process(mask, patn):
    mask_tanh = torch.tanh(mask) / (2 - epsilon()) + 0.5
    patn_tanh = torch.tanh(patn) / (2 - epsilon()) + 0.5
    return mask_tanh, patn_tanh


def nc(model, x, y_target, preprocess):
    # Initialize the mask and pattern
    mask_init = np.random.random((1, 1, 32, 32))
    patn_init = np.random.random((1, 3, 32, 32))

    mask_init = np.arctanh((mask_init - 0.5) * (2 - epsilon()))
    patn_init = np.arctanh((patn_init - 0.5) * (2 - epsilon()))

    # Define optimizing parameters
    mask = torch.FloatTensor(mask_init).cuda()
    mask.requires_grad = True
    patn = torch.FloatTensor(patn_init).cuda()
    patn.requires_grad = True

    # Define the optimization
    optimizer = torch.optim.Adam(params=[mask, patn], lr=1e-1, betas=(0.5, 0.9))

    # Loss cnn and weights
    criterion = nn.CrossEntropyLoss()

    reg_best = 1 / epsilon()

    # Threshold for attack success rate
    init_asr_threshold = 0.99
    asr_threshold = init_asr_threshold

    # Initial cost for regularization
    init_cost = 1e-3
    cost = init_cost
    cost_multiplier_up = 2
    cost_multiplier_down = cost_multiplier_up ** 1.5

    # Counters for adjusting balance cost
    cost_set_counter = 0
    cost_up_counter = 0
    cost_down_counter = 0
    cost_up_flag = False
    cost_down_flag = False

    # Counter for early stop
    early_stop = True
    early_stop_threshold = 1.0
    early_stop_counter = 0
    early_stop_reg_best = reg_best

    # Patience
    patience = 5
    early_stop_patience = 5 * patience
    threshold_patience = patience

    # Total optimization steps
    steps = 1000
    
    # Start optimization
    for step in range(steps):
        mask_tanh, patn_tanh = mask_patn_process(mask, patn)

        px = (1 - mask_tanh) * x + mask_tanh * patn_tanh

        input_x = preprocess(px)
        input_y = torch.zeros(input_x.shape[0], dtype=torch.long).cuda() + y_target
        logits = model(input_x)
        ce_loss = criterion(logits, input_y)

        reg_loss = torch.sum(mask_tanh)
        loss = ce_loss + cost * reg_loss

        CE_LOSS = ce_loss.sum().item()
        REG_LOSS = reg_loss.sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = logits.max(dim=1)[1]
        n_asr = (pred == y_target).sum().item()
        asr = n_asr / pred.shape[0]

        if step % 10 == 0:
            print(f'Y_target: {y_target}, Step: {step}, cost: {cost:.4f}, CE_LOSS: {CE_LOSS:.4f}, REG_LOSS: {REG_LOSS:.4f}, ASR: {asr*100:.2f}%')

        if asr >= asr_threshold and REG_LOSS < reg_best:
            px_best = px.detach().cpu()
            savefig = torch.cat([torch.repeat_interleave(mask_tanh, 3, dim=1), patn_tanh, patn_tanh * mask_tanh], dim=0).detach().cpu()
            reg_best = REG_LOSS
        
        # Check early stop
        if early_stop:
            # Only terminate if a valid attack has been found
            if reg_best < 1 / epsilon():
                if reg_best >= early_stop_threshold * early_stop_reg_best:
                    early_stop_counter += 1
                else:
                    early_stop_counter = 0
            early_stop_reg_best = min(reg_best, early_stop_reg_best)

            if (cost_down_flag and cost_up_flag and early_stop_counter >= early_stop_patience):
                print('Early stop !\n')
                break
        
        # Check cost modification
        if cost < epsilon() and asr >= asr_threshold:
            cost_set_counter += 1
            if cost_set_counter >= threshold_patience:
                cost = init_cost
                cost_up_counter = 0
                cost_down_counter = 0
                cost_up_flag = False
                cost_down_flag = False
                print('*** Initialize cost to %.2E' % (cost))
        else:
            cost_set_counter = 0
        
        if asr >= asr_threshold:
            cost_up_counter += 1
            cost_down_counter = 0
        else:
            cost_up_counter = 0
            cost_down_counter += 1
        
        if cost_up_counter >= patience:
            cost_up_counter = 0
            cost *= cost_multiplier_up
            cost_up_flag = True
            # print('UP cost to %.2E' % cost)
        if cost_down_counter >= patience:
            cost_down_counter = 0
            cost /= cost_multiplier_down
            cost_down_flag = True
            # print('DOWN cost to %.2E' % cost)
    
    return reg_best, px_best, savefig


def dual_tanh(model, x, y_target, preprocess):
    # Define optimizing parameters
    pattern_shape = (1, 3, 32, 32)
    for i in range(2):
        init_pattern = np.random.random(pattern_shape)
        init_pattern = np.clip(init_pattern, 0.0, 1.0)

        if i == 0:
            pattern_pos_tensor = torch.Tensor(init_pattern).cuda()
            pattern_pos_tensor.requires_grad = True
        else:
            pattern_neg_tensor = torch.Tensor(- init_pattern).cuda()
            pattern_neg_tensor.requires_grad = True

    # Define the optimization
    optimizer = torch.optim.Adam(params=[pattern_pos_tensor, pattern_neg_tensor], lr=1e-1, betas=(0.5, 0.9))

    # Loss cnn and weights
    criterion = nn.CrossEntropyLoss()

    reg_best = 1 / epsilon()
    pixel_best = 1 / epsilon()

    # Threshold for attack success rate
    init_asr_threshold = 0.9
    asr_threshold = init_asr_threshold

    # Initial cost for regularization
    init_cost = 1e-3
    cost = init_cost
    cost_multiplier_up = 1.5
    cost_multiplier_down = cost_multiplier_up ** 1.5

    # Counters for adjusting balance cost
    cost_up_counter = 0
    cost_down_counter = 0

    # Patience
    patience = 10

    # Total optimization steps
    steps = 1000
    
    # Start optimization
    for step in range(steps):
        pattern_pos = torch.clamp(pattern_pos_tensor, 0.0, 1.0)
        pattern_neg = - torch.clamp(pattern_neg_tensor, 0.0, 1.0)

        px = x + pattern_pos + pattern_neg
        px = torch.clamp(px, 0.0, 1.0)

        input_x = preprocess(px)
        input_y = torch.zeros(input_x.shape[0], dtype=torch.long).cuda() + y_target
        logits = model(input_x)
        ce_loss = criterion(logits, input_y)
        reg_pos  = torch.max(torch.tanh(pattern_pos_tensor / 10) / (2 - epsilon()) + 0.5, axis=0)[0]
        reg_neg  = torch.max(torch.tanh(pattern_neg_tensor / 10) / (2 - epsilon()) + 0.5, axis=0)[0]
        reg_loss = torch.sum(reg_pos) + torch.sum(reg_neg)
        
        loss = ce_loss + cost * reg_loss
        
        CE_LOSS = ce_loss.sum().item()
        REG_LOSS = reg_loss.sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = logits.max(dim=1)[1]
        n_asr = (pred == y_target).sum().item()
        asr = n_asr / pred.shape[0]

        if step % 10 == 0:
            print(f'Y_target: {y_target}, Step: {step}, cost: {cost:.4f}, CE_LOSS: {CE_LOSS:.4f}, REG_LOSS: {REG_LOSS:.4f}, ASR: {asr * 100:.2f}%')
        
        # remove small pattern values
        threshold = 1.0 / 255.0
        pattern_pos_cur = pattern_pos.detach()
        pattern_neg_cur = pattern_neg.detach()
        pattern_pos_cur[(pattern_pos_cur < threshold) & (pattern_pos_cur > -threshold)] = 0
        pattern_neg_cur[(pattern_neg_cur < threshold) & (pattern_neg_cur > -threshold)] = 0
        pattern_cur = pattern_pos_cur + pattern_neg_cur

        # count current number of perturbed pixels
        pixel_cur = np.count_nonzero(np.sum(np.abs(pattern_cur.cpu().numpy()), axis=0))

        # Record the best pattern
        if asr >= asr_threshold and REG_LOSS < reg_best and pixel_cur < pixel_best:
            pattern_pos_best = pattern_pos.detach()
            pattern_pos_best[pattern_pos_best < threshold] = 0
            init_pattern = pattern_pos_best
            with torch.no_grad():
                pattern_pos_tensor.copy_(init_pattern)

            pattern_neg_best = pattern_neg.detach()
            pattern_neg_best[pattern_neg_best > -threshold] = 0
            init_pattern = - pattern_neg_best
            with torch.no_grad():
                pattern_neg_tensor.copy_(init_pattern)

            pattern_best = pattern_pos_best + pattern_neg_best
            px_best = torch.clamp(x + pattern_best, 0.0, 1.0).detach().cpu()
            savefig = pattern_best.detach().cpu()

            reg_best = REG_LOSS
            pixel_best = pixel_cur

            best_size = np.count_nonzero(pattern_best.abs().sum(0).cpu().numpy())
        
        # helper variables for adjusting loss weight
        if asr >= asr_threshold:
            cost_up_counter += 1
            cost_down_counter = 0
        else:
            cost_up_counter = 0
            cost_down_counter += 1

        # adjust loss weight
        if cost_up_counter >= patience:
            cost_up_counter = 0
            if cost == 0:
                cost = init_cost
            else:
                cost *= cost_multiplier_up
        elif cost_down_counter >= patience:
            cost_down_counter = 0
            cost /= cost_multiplier_down
    
    if reg_best == 1 / epsilon():
        pattern_pos_best = pattern_pos.detach()
        pattern_pos_best[pattern_pos_best < threshold] = 0
        init_pattern = pattern_pos_best
        with torch.no_grad():
            pattern_pos_tensor.copy_(init_pattern)

        pattern_neg_best = pattern_neg.detach()
        pattern_neg_best[pattern_neg_best > -threshold] = 0
        init_pattern = - pattern_neg_best
        with torch.no_grad():
            pattern_neg_tensor.copy_(init_pattern)

        pattern_best = pattern_pos_best + pattern_neg_best
        px_best = torch.clamp(x + pattern_best, 0.0, 1.0).detach().cpu()
        savefig = pattern_best.detach().cpu()

        reg_best = REG_LOSS
        pixel_best = pixel_cur

        best_size = np.count_nonzero(pattern_best.abs().sum(0).cpu().numpy())
    
    return best_size, px_best, savefig


def outlier_detection(l1_norm_list):
    consistency_constant = 1.4826  # if normal distribution
    median = np.median(l1_norm_list)
    mad = consistency_constant * np.median(np.abs(l1_norm_list - median))
    min_mad = np.abs(np.min(l1_norm_list) - median) / mad

    print('median: %f, MAD: %f' % (median, mad))
    print('anomaly index: %f' % min_mad)

    flag_list = []
    num_classes = len(l1_norm_list)
    for y_label in range(num_classes):
        if l1_norm_list[y_label] > median:
            continue
        if np.abs(l1_norm_list[y_label] - median) / mad > 2:
            flag_list.append((y_label, l1_norm_list[y_label]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])
    
    print('flagged label list: %s' % ', '.join(['%d: %.2f' % (y_label, l_norm) for y_label, l_norm in flag_list]))

    return min_mad


def load_samples(examples_dirpath):
    dataset = pickle.load(open(examples_dirpath, 'rb'), encoding='bytes')
    fxs, fys = dataset['x_val'], dataset['y_val']
    fxs, fys = np.uint8(fxs), np.asarray(fys).astype(np.int)
    assert(fxs.shape[0] == 100)
    assert(fys.shape[0] == 100)

    # print('number of seed images', fxs.shape, fys.shape)
    return fxs, fys


def main(model_filepath, examples_filepath, phase, preprocess, num_classes):
    # Load model
    model = torch.load(model_filepath).cuda()
    model.eval()

    # Get validation samples
    fxs, _ = load_samples(examples_filepath)
    test_samples = torch.from_numpy(fxs / 255.0).permute(0, 3, 1, 2).float().cuda()

    # Perform detection
    l1_norm_list = []
    for y_target in range(num_classes):
        print(f'Y_target: {y_target}')
        if phase == 'nc':
            best_size, px_best, savefig = nc(model, test_samples, y_target, preprocess)
        elif phase == 'dual_tanh':
            best_size, px_best, savefig = dual_tanh(model, test_samples, y_target, preprocess)
        else:
            raise NotImplementedError

        # Save results
        # save_dir = f'detection/{args.phase}/{args.attack}_{args.par}'
        # os.makedirs(save_dir, exist_ok=True)
        # save_image(px_best[:8], f'{save_dir}/images_{args.victim}_{y_target}.png')
        # save_image(savefig, f'{save_dir}/triggers_{args.victim}_{y_target}.png')

        l1_norm_list.append(best_size)
    
    # Outlier detection
    anomaly_index = outlier_detection(l1_norm_list)
    return anomaly_index
