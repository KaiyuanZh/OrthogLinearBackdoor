import os
import sys
import time
import cv2
import math
import random
import pickle
import argparse
import numpy as np

import torch
import torch.nn.functional as F


np.set_printoptions(precision=2, linewidth=200, threshold=10000)


config = {}
config['print_level'] = 1
config['random_seed'] = 333
config['channel_last'] = 0
config['w'] = 32
config['h'] = 32
config['reasr_bound'] = 0.2
config['batch_size'] = 10
config['has_softmax'] = 0
config['samp_k'] = 8
config['same_range'] = 0
config['n_samples'] = 5
config['samp_batch_size'] = 1
config['top_n_neurons'] = 10
config['re_batch_size'] = 80

config['max_troj_size'] = 64

config['filter_multi_start'] = 1
config['re_mask_lr'] = 4e-2
config['re_mask_weight'] = 5000
config['mask_multi_start'] = 1
config['re_epochs'] = 50
config['n_re_samples'] = 240

channel_last = bool(config['channel_last'])

resnet_sample_resblock = False

# deterministic
random_seed = config['random_seed']
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

w = config["w"]
h = config["h"]
num_classes = 10
use_mask = True
count_mask = True
tdname = 'temp'
window_size = 12
mask_epsilon = 0.01
mask_epsilon = 0.1
delta_shape = [window_size,window_size,3,3]
Troj_size = config['max_troj_size']
reasr_bound = float(config['reasr_bound'])
top_n_neurons = int(config['top_n_neurons'])
mask_multi_start = int(config['mask_multi_start'])
filter_multi_start = int(config['filter_multi_start'])
re_mask_weight = float(config['re_mask_weight'])
re_mask_lr = float(config['re_mask_lr'])
batch_size = config['batch_size']
has_softmax = bool(config['has_softmax'])
# print('channel_last', channel_last, 'has softmax', has_softmax)

Print_Level = int(config['print_level'])
re_epochs = int(config['re_epochs'])
n_re_samples = int(config['n_re_samples'])

# Pre-processing
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])
l_bounds = np.asarray([(0.0 - mean[0]) / std[0], (0.0 - mean[1]) / std[1], (0.0 - mean[2]) / std[2]])
h_bounds = np.asarray([(1.0 - mean[0]) / std[0], (1.0 - mean[1]) / std[1], (1.0 - mean[2]) / std[2]])
l_bounds_tensor = torch.FloatTensor(l_bounds).cuda()
h_bounds_tensor = torch.FloatTensor(h_bounds).cuda()


def preprocess(img):
    img = np.transpose(img, [0, 3, 1, 2])
    return img.astype(np.float32) / 255.0


def deprocess(x_in):
    x_in = x_in * std.reshape((1, 3, 1, 1)) + mean.reshape((1, 3, 1, 1))
    x_in *= 255
    return x_in.astype('uint8')


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


def check_values(images, labels, model, children, target_layers):
    maxes = {}
    for layer_i in range(0, len(children) - 1):
        if not children[layer_i].__class__.__name__ in target_layers:
            continue
        temp_model1 = torch.nn.Sequential(*children[:layer_i+1])

        max_val = -np.inf
        for i in range( math.ceil(float(len(images))/batch_size) ):
            batch_data = torch.FloatTensor(images[batch_size*i:batch_size*(i+1)])
            batch_data = batch_data.cuda()
            inner_outputs = temp_model1(batch_data).cpu().detach().numpy()
            if channel_last:
                n_neurons = inner_outputs.shape[-1]
            else:
                n_neurons = inner_outputs.shape[1]
            max_val = np.maximum(max_val, np.amax(inner_outputs))
            # print(np.amax(inner_outputs))
        
        key = '{0}_{1}'.format(children[layer_i].__class__.__name__, layer_i)
        maxes[key] = [max_val]
        # print('max val', key, max_val)
        del temp_model1, batch_data, inner_outputs
    return maxes


def sample_neuron(images, labels, model, children, target_layers, model_type, mvs, has_softmax=has_softmax):
    all_ps = {}
    samp_k = config['samp_k']
    same_range = config['same_range']
    n_samples = config['n_samples']
    sample_batch_size = config['samp_batch_size']
    if model_type == 'DenseNet':
        sample_batch_size = max(sample_batch_size // 3, 1)
    n_images = images.shape[0]
    if Print_Level > 0:
        print('sampling n imgs', n_images)

    end_layer = len(children)-1
    if has_softmax:
        end_layer = len(children)-2

    sample_layers = []
    for layer_i in range(2, end_layer):
        if not children[layer_i].__class__.__name__ in target_layers:
            continue
        sample_layers.append(layer_i)
    
    # TODO:
    sample_layers = sample_layers[-1:]

    for layer_i in sample_layers:
        if Print_Level > 0:
            print('layer', layer_i, children[layer_i])
        temp_model1 = torch.nn.Sequential(*children[:layer_i+1])
        if has_softmax:
            temp_model2 = torch.nn.Sequential(*children[layer_i+1:-1])
        else:
            temp_model2 = torch.nn.Sequential(*children[layer_i+1:])

        if same_range:
            vs = np.asarray([i*samp_k for i in range(n_samples)])
        else:
            mv_key = '{0}_{1}'.format(children[layer_i].__class__.__name__, layer_i)

            tr = samp_k * max(mvs[mv_key])/(n_samples)
            vs = np.asarray([i*tr for i in range(n_samples)])
        
        for input_i in range( math.ceil(float(n_images)/batch_size) ):
            cbatch_size = min(batch_size, n_images - input_i*batch_size)
            # print('batch_size', batch_size, 'cbatch_size', cbatch_size, )
            batch_data = torch.FloatTensor(images[batch_size*input_i:batch_size*(input_i+1)])
            batch_data = batch_data.cuda()
            inner_outputs = temp_model1(batch_data).cpu().detach().numpy()
            n_neurons = inner_outputs.shape[1]

            nbatches = math.ceil(float(n_neurons)/sample_batch_size)
            for nt in range(nbatches):
                l_h_t = []
                csample_batch_size = min(sample_batch_size, n_neurons - nt*sample_batch_size)
                for neuron in range(csample_batch_size):
                    if len(inner_outputs.shape) == 4:
                        h_t = np.tile(inner_outputs, (n_samples, 1, 1, 1))
                    else:
                        h_t = np.tile(inner_outputs, (n_samples, 1))

                    for i,v in enumerate(vs):
                        h_t[i*cbatch_size:(i+1)*cbatch_size,neuron+nt*sample_batch_size,:,:] = v
                    l_h_t.append(h_t)
                f_h_t = np.concatenate(l_h_t, axis=0)

                f_h_t_t = torch.FloatTensor(f_h_t).cuda()
                fps = temp_model2( f_h_t_t ).cpu().detach().numpy()
                for neuron in range(csample_batch_size):
                    tps = fps[neuron*n_samples*cbatch_size:(neuron+1)*n_samples*cbatch_size]

                    for img_i in range(cbatch_size):
                        img_name = (labels[img_i + batch_size*input_i], img_i + batch_size*input_i)
                        ps_key= (img_name, '{0}_{1}'.format(children[layer_i].__class__.__name__, layer_i), neuron+nt*sample_batch_size)
                        ps = [tps[ img_i +cbatch_size*_] for _ in range(n_samples)]
                        ps = np.asarray(ps)
                        ps = ps.T
                        # print('img i', img_i, input_i, batch_size, 'neuron', neuron, ps_key, ps.shape)
                        all_ps[ps_key] = np.copy(ps)
                
                del f_h_t_t
            del batch_data, inner_outputs
            torch.cuda.empty_cache()

        del temp_model1, temp_model2
    return all_ps, sample_layers


def find_min_max(model_name, all_ps, sample_layers, cut_val=20, top_k=10):
    max_ps = {}
    max_vals = []
    n_classes = 0
    n_samples = 0
    for k in sorted(all_ps.keys()):
        all_ps[k] = all_ps[k][:, :cut_val]
        n_classes = all_ps[k].shape[0]
        n_samples = all_ps[k].shape[1]
        # maximum increase diff

        vs = []
        for l in range(num_classes):
            vs.append( np.amax(all_ps[k][l][1:]) - np.amin(all_ps[k][l][:1]) )
        ml = np.argsort(np.asarray(vs))[-1]
        sml = np.argsort(np.asarray(vs))[-2]
        val = vs[ml] - vs[sml]

        max_vals.append(val)
        max_ps[k] = (ml, val)
    
    neuron_ks = []
    imgs = []
    for k in sorted(max_ps.keys()):
        nk = (k[1], k[2])
        neuron_ks.append(nk)
        imgs.append(k[0])
    neuron_ks = list(set(neuron_ks))
    imgs = list(set(imgs))
    
    min_ps = {}
    min_vals = []
    n_imgs = len(imgs)
    for k in neuron_ks:
        vs = []
        ls = []
        vdict = {}
        for img in sorted(imgs):
            nk = (img, k[0], k[1])
            l = max_ps[nk][0]
            v = max_ps[nk][1]
            vs.append(v)
            ls.append(l)
            if not ( l in vdict.keys() ):
                vdict[l] = [v]
            else:
                vdict[l].append(v)
        ml = max(set(ls), key=ls.count)


        fvs = []
        # does not count when l not equal ml
        for img in sorted(imgs):
            img_l = int(img[0])
            nk = (img, k[0], k[1])
            l = max_ps[nk][0]
            v = max_ps[nk][1]
            if l != ml:
                continue
            fvs.append(v)
        
        if len(fvs) > 0:
            min_ps[k] = (ml, ls.count(ml), np.mean(fvs), fvs)
            min_vals.append(np.mean(fvs))

        else:
            min_ps[k] = (ml, 0, 0, fvs)
            min_vals.append(0)
    
    keys = min_ps.keys()
    keys = []
    for k in min_ps.keys():
        if min_ps[k][1] >= int(n_imgs * 0.6):
            keys.append(k)
    if len(keys) == 0:
        for k in min_ps.keys():
            if min_ps[k][1] >= int(n_imgs * 0.1):
                keys.append(k)
    sorted_key = sorted(keys, key=lambda x: min_ps[x][2] )
    if Print_Level > 0:
        print('n samples', n_samples, 'n class', n_classes, 'n_imgs', n_imgs)

    neuron_dict = {}
    neuron_dict[model_name] = []
    maxval = min_ps[sorted_key[-1]][2]
    layers = {}
    labels = {}
    allns = 0
    max_sampling_val = -np.inf

    # last layers
    labels = {}
    for i in range(len(sorted_key)):
        k = sorted_key[-i-1]
        layer = k[0]
        neuron = k[1]
        label = min_ps[k][0]
        if (layer, neuron, min_ps[k][0]) in neuron_dict[model_name]:
            continue

        if label not in labels.keys():
            labels[label] = 0
        # if int(layer.split('_')[-1]) == sample_layers[-1] and labels[label] < 1:
        if True:
            labels[label] += 1

            if min_ps[k][2] > max_sampling_val:
                max_sampling_val = min_ps[k][2]
            if Print_Level > 0:
                print(i, 'min max val across images', 'k', k, 'label', min_ps[k][0], min_ps[k][1], 'value', min_ps[k][2])
                if Print_Level > 1:
                    print(min_ps[k][3])
            allns += 1
            neuron_dict[model_name].append( (layer, neuron, min_ps[k][0]) )
        if allns >= top_k:
            break

    return neuron_dict, max_sampling_val


def read_all_ps(model_name, all_ps, sample_layers, top_k=10, cut_val=20):
    return find_min_max(model_name, all_ps, sample_layers,  cut_val, top_k=top_k)


def filter_img():
    mask = np.zeros((h, w), dtype=np.float32)
    Troj_w = int(np.sqrt(Troj_size) * 0.8)
    for i in range(h):
        for j in range(w):
            # if j >= h/2 and j < h/2 + Troj_w and i >= w/2 and i < w/2 + Troj_w:
            if j < Troj_w and i < Troj_w:
                mask[j, i] = 1
    return mask


def nc_filter_img():
    if use_mask:
        mask = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                if not( j >= w*1/4.0 and j < w*3/4.0 and i >= h*1/4.0 and i < h*3/4.0):
                    mask[i, j] = 1
    else:
        mask = np.zeros((h, w), dtype=np.float32) + 1
    return mask


def loss_fn(inner_outputs_b, inner_outputs_a, logits, con_mask, neuron, tlabel, acc, e, re_epochs):
    neuron_mask = torch.zeros([1, inner_outputs_a.shape[1],1,1]).cuda()
    neuron_mask[:,neuron,:,:] = 1
    vloss1     = torch.sum(inner_outputs_b * neuron_mask)/torch.sum(neuron_mask)
    vloss2     = torch.sum(inner_outputs_b * (1-neuron_mask))/torch.sum(1-neuron_mask)
    relu_loss1 = torch.sum(inner_outputs_a * neuron_mask)/torch.sum(neuron_mask)
    relu_loss2 = torch.sum(inner_outputs_a * (1-neuron_mask))/torch.sum(1-neuron_mask)

    vloss3     = torch.sum(inner_outputs_b * torch.lt(inner_outputs_b, 0) )/torch.sum(1-neuron_mask)

    loss = - vloss1 - relu_loss1  + 0.0001 * vloss2 + 0.0001 * relu_loss2
    mask_loss = torch.sum(con_mask)
    mask_nz = torch.sum(torch.gt(con_mask, mask_epsilon))
    mask_cond1 = torch.gt(mask_nz, Troj_size)
    mask_cond2 = torch.gt(mask_nz, Troj_size * 1.2)
    mask_add_loss = torch.where(mask_cond1, torch.where(mask_cond2, 10 * re_mask_weight * mask_loss, 5 * re_mask_weight * mask_loss), 0.0 * mask_loss)
    loss += mask_add_loss
    logits_loss = torch.sum(logits[:,tlabel]) - 0.001 * ( torch.sum(logits[:,:tlabel]) + torch.sum(logits[:,tlabel:]) )
    loss += - 2e2 * logits_loss
    return loss, vloss1, vloss2, vloss3, relu_loss1, relu_loss2, mask_loss, mask_nz, mask_add_loss, logits_loss


def reverse_engineer(model_type, model, children, oimages, olabels, weights_file, Troj_Layer, Troj_Neuron, Troj_Label, Troj_size, re_epochs):
    
    before_block = []
    def get_before_block():
        def hook(model, input, output):
            for ip in input:
                before_block.append( ip.clone() )
        return hook
    
    after_bn3 = []
    def get_after_bn3():
        def hook(model, input, output):
            for ip in output:
                after_bn3.append( ip.clone() )
        return hook
    
    after_iden = []
    def get_after_iden():
        def hook(model, input, output):
            for ip in output:
                after_iden.append( ip.clone() )
        return hook

    after_bns = []
    def get_after_bns():
        def hook(model, input, output):
            for ip in output:
                after_bns.append( ip.clone() )
        return hook


    re_batch_size = config['re_batch_size']
    if model_type in ['ResNet', 'PreActResNet', 'WideResNet']:
        re_batch_size = max(re_batch_size // 4, 1)
    if model_type == 'VGG':
        re_batch_size = max(re_batch_size // 4, 1)
    if re_batch_size > len(oimages):
        re_batch_size = len(oimages)

    handles = []
    if model_type == 'VGG':
        tmodule1 = children[Troj_Layer]
        handle = tmodule1.register_forward_hook(get_after_bns())
        handles.append(handle)
    elif model_type in ['ResNet', 'PreActResNet', 'WideResNet']:
        if resnet_sample_resblock:
            children_modules = list(children[Troj_Layer].children())
        else:
            children_modules = list(list(children[Troj_Layer].children())[-1].children())
        # print(len(children_modules), children_modules)

        last_bn_id = 0
        has_downsample = False
        i = 0
        for children_module in children_modules:
            if children_module.__class__.__name__ == 'BatchNorm2d':
                last_bn_id = i
            if children_module.__class__.__name__ == 'Sequential':
                has_downsample = True
            i += 1
        # print('last bn id', last_bn_id, 'has_downsample', has_downsample)
        bn3_module = children_modules[last_bn_id]
        handle = bn3_module.register_forward_hook(get_after_bn3())
        handles.append(handle)
        if has_downsample:
            iden_module = children_modules[-1]
            handle = iden_module.register_forward_hook(get_after_iden())
            handles.append(handle)
        else:
            iden_module = children_modules[0]
            handle = iden_module.register_forward_hook(get_before_block())
            handles.append(handle)

    # print('Target Layer', Troj_Layer, children[Troj_Layer], 'Neuron', Troj_Neuron, 'Target Label', Troj_Label)

    # delta = torch.randn(1, 3, h, w).cuda()
    delta = torch.rand(1, 3, h, w).cuda() * 2 - 1
    # TODO: Random initialization
    mask = filter_img().reshape((1, 1, h, w)) * 8 - 4
    # mask = torch.ones(1, 1, h, w) * 8 - 4
    mask= torch.FloatTensor(mask).cuda()
    delta.requires_grad = True
    mask.requires_grad = True
    optimizer = torch.optim.Adam([delta, mask], lr=re_mask_lr)
    # print('before optimizing',)
    for e in range(re_epochs):
        facc = 0
        flogits = []
        p = np.random.permutation(oimages.shape[0])
        images = oimages[p]
        labels = olabels[p]
        for i in range( math.ceil(float(len(images))/re_batch_size) ):
            cre_batch_size = min(len(images) - re_batch_size * i, re_batch_size)
            optimizer.zero_grad()
            model.zero_grad()
            after_bn3.clear()
            before_block.clear()
            after_iden.clear()
            after_bns.clear()

            batch_data = torch.FloatTensor(images[re_batch_size*i:re_batch_size*(i+1)])
            batch_data = batch_data.cuda()

            con_mask = torch.tanh(mask)/2.0 + 0.5
            con_delta = torch.tanh(delta)/2.0 + 0.5
            use_delta = (con_delta - torch.FloatTensor(mean.reshape(1,3,1,1)).cuda() )/ torch.FloatTensor(std.reshape(1,3,1,1)).cuda()
            use_mask = con_mask
            in_data = use_mask * use_delta + (1-use_mask) * batch_data

            clamp = [True, False][0]
            # Batch data is clipped in [l_bounds, h_bounds]
            if clamp:
                batch_r = torch.clamp(in_data[:, 0, :, :], min=l_bounds_tensor[0], max=h_bounds_tensor[0])
                batch_g = torch.clamp(in_data[:, 1, :, :], min=l_bounds_tensor[1], max=h_bounds_tensor[1])
                batch_b = torch.clamp(in_data[:, 2, :, :], min=l_bounds_tensor[2], max=h_bounds_tensor[2])
                in_data = torch.stack([batch_r, batch_g, batch_b], dim=1)
            
            logits = model(in_data)
            logits_np = logits.cpu().detach().numpy()
            
            if model_type == 'VGG':
                inner_outputs_b = torch.stack(after_bns, 0)
                inner_outputs_a = F.relu(inner_outputs_b)
            elif model_type in ['ResNet', 'PreActResNet', 'WideResNet']:
                after_bn3_t = torch.stack(after_bn3, 0)
                iden = None
                if len(before_block) > 0:
                    iden = before_block[0]
                else:
                    after_iden_t = torch.stack(after_iden, 0)
                    iden = after_iden_t
                
                # TODO: Problematic
                # inner_outputs_b = iden + after_bn3_t
                inner_outputs_b = after_bn3_t
                # print(iden.shape, after_bn3_t.shape, iden.dtype, after_bn3_t.dtype)
                inner_outputs_a = F.relu(inner_outputs_b)

            flogits.append(logits_np)
            loss, vloss1, vloss2, vloss3, relu_loss1, relu_loss2, mask_loss, mask_nz, mask_add_loss, logits_loss\
                    = loss_fn(inner_outputs_b, inner_outputs_a, logits, use_mask, Troj_Neuron, int(Troj_Label), facc, e, re_epochs)
            if e > 0:
                loss.backward(retain_graph=True)
                optimizer.step()
            # break
        flogits = np.concatenate(flogits, axis=0)
        preds = np.argmax(flogits, axis=1)

        # do not change Troj_Label
        # Troj_Label2 = np.argmax(np.bincount(preds))
        Troj_Label2 = Troj_Label

        facc = np.sum(preds == Troj_Label2) / float(preds.shape[0])

        if e % 10 == 0 and Print_Level > 0:
            print(e, 'loss', loss.cpu().detach().numpy(), 'acc {:.4f}'.format(facc),'target label', int(Troj_Label), int(Troj_Label2), 'logits_loss', logits_loss.cpu().detach().numpy(),\
                    'vloss1', vloss1.cpu().detach().numpy(), 'vloss2', vloss2.cpu().detach().numpy(),\
                    'relu_loss1', relu_loss1.cpu().detach().numpy(), 'max relu_loss1', np.amax(inner_outputs_a.cpu().detach().numpy()),\
                    'relu_loss2', relu_loss2.cpu().detach().numpy(),\
                    'mask_loss', mask_loss.cpu().detach().numpy(), 'mask_nz', mask_nz.cpu().detach().numpy(), 'mask_add_loss', mask_add_loss.cpu().detach().numpy())
            print('labels', flogits[:5,:10])
            print('logits', np.argmax(flogits, axis=1))
            print('delta', use_delta[0,0,:5,:5])
            print('mask', use_mask[0,0,:5,:5])

        # if facc > 0.99:
        #     break
    delta = use_delta.cpu().detach().numpy()
    con_mask = use_mask.cpu().detach().numpy()
    adv = in_data.cpu().detach().numpy()
    # adv = deprocess(adv)

    # cleaning up
    for handle in handles:
        handle.remove()

    return facc, adv, delta, con_mask, Troj_Label2


def re_mask(model_type, model, neuron_dict, children, images, labels, scratch_dirpath, re_epochs):
    validated_results = []
    for key in sorted(neuron_dict.keys()):
        weights_file = key
        for task in neuron_dict[key]:
            Troj_Layer, Troj_Neuron, samp_label = task
            Troj_Neuron = int(Troj_Neuron)
            Troj_Layer = int(Troj_Layer.split('_')[1])

            RE_img = os.path.join(scratch_dirpath,'imgs', '{0}_model_{1}_{2}_{3}_{4}.png'.format(    weights_file.split('/')[-1].split('\.')[0], Troj_Layer, Troj_Neuron, Troj_size, samp_label))
            RE_mask = os.path.join(scratch_dirpath,'masks', '{0}_model_{1}_{2}_{3}_{4}.pkl'.format(  weights_file.split('/')[-1].split('\.')[0], Troj_Layer, Troj_Neuron, Troj_size, samp_label))
            RE_delta = os.path.join(scratch_dirpath,'deltas', '{0}_model_{1}_{2}_{3}_{4}.pkl'.format(weights_file.split('/')[-1].split('\.')[0], Troj_Layer, Troj_Neuron, Troj_size, samp_label))
            
            max_acc = 0
            max_results = []
            for i  in range(mask_multi_start):
                acc, rimg, rdelta, rmask, optz_label = reverse_engineer(model_type, model, children, images, labels, weights_file, Troj_Layer, Troj_Neuron, samp_label, Troj_size, re_epochs)

                # clear cache
                torch.cuda.empty_cache()

                if Print_Level > 0:
                    print('RE mask', Troj_Layer, Troj_Neuron, 'Label', optz_label, 'RE acc', acc)
                if acc > max_acc:
                    max_acc = acc
                    max_results = (rimg, rdelta, rmask, optz_label, RE_img, RE_mask, RE_delta, samp_label, acc)
            if max_acc >= reasr_bound - 0.2:
                validated_results.append( max_results )
            # if max_acc > 0.99 and optz_label == samp_label:
            #     break

    return validated_results


def stamp(n_img, delta, mask):
    mask0 = nc_filter_img()
    mask = mask * mask0
    r_img = n_img.copy()
    mask = mask.reshape((1,1,w,h))
    # print('stamp', np.amax(n_img), np.amin(n_img), np.amax(delta), np.amin(delta), np.sum(mask), np.amax(mask), np.amin(mask))
    r_img = n_img * (1-mask) + delta * mask
    return r_img


def test(model, model_type, test_xs, result, scratch_dirpath, mode='mask'):
    
    re_batch_size = config['re_batch_size']
    if model_type in ['ResNet', 'PreActResNet']:
        re_batch_size = max(re_batch_size // 4, 1)
    if model_type == 'VGG':
        re_batch_size = max(re_batch_size // 4, 1)
    if re_batch_size > len(test_xs):
        re_batch_size = len(test_xs)

    clean_images = test_xs

    rimg, rdelta, rmask, tlabel, RE_img = result[:5]
    rmask = rmask * rmask > mask_epsilon
    t_images = stamp(clean_images, rdelta, rmask)

    saved_images = deprocess(t_images)
    
    rt_images = t_images
    if Print_Level > 0:
        print(np.amin(rt_images), np.amax(rt_images))
    
    yt = np.zeros(len(rt_images)).astype(np.int32) + tlabel
    fpreds = []
    for i in range( math.ceil(float(len(rt_images))/re_batch_size) ):
        batch_data = torch.FloatTensor(rt_images[re_batch_size*i:re_batch_size*(i+1)])
        batch_data = batch_data.cuda()
        preds = model(batch_data)
        fpreds.append(preds.cpu().detach().numpy())
    fpreds = np.concatenate(fpreds)

    preds = np.argmax(fpreds, axis=1) 
    # print(preds)
    score = float(np.sum(tlabel == preds))/float(yt.shape[0])
    top5_preds = np.argsort(fpreds, axis=1)[:,-5:]
    top5_acc = np.sum(np.any(top5_preds == yt[:, np.newaxis],axis=1)) / float(yt.shape[0])
    # print('label', tlabel, 'score', score)
    return score, top5_acc


def load_samples(examples_dirpath):
    dataset = pickle.load(open(examples_dirpath, 'rb'), encoding='bytes')
    fxs, fys = dataset['x_val'], dataset['y_val']
    fxs, fys = np.uint8(fxs), np.asarray(fys).astype(np.int)
    assert(fxs.shape[0] == 100)
    assert(fys.shape[0] == 100)

    print(fxs.shape, fys.shape)
    print(fxs.max(), fxs.min(), fxs.mean(), fxs.std())

    # print('number of seed images', fxs.shape, fys.shape)
    return fxs, fys


def main(model_filepath, scratch_dirpath, examples_dirpath, example_img_format='png'):
    start = time.time()

    # create dirs
    os.system('mkdir -p {0}'.format(os.path.join(scratch_dirpath, 'imgs')))
    os.system('mkdir -p {0}'.format(os.path.join(scratch_dirpath, 'masks')))
    os.system('mkdir -p {0}'.format(os.path.join(scratch_dirpath, 'temps')))
    os.system('mkdir -p {0}'.format(os.path.join(scratch_dirpath, 'deltas')))

    # remove previous results
    os.system('rm -r {0}/*'.format(os.path.join(scratch_dirpath, 'imgs')))
    os.system('rm -r {0}/*'.format(os.path.join(scratch_dirpath, 'masks')))
    os.system('rm -r {0}/*'.format(os.path.join(scratch_dirpath, 'temps')))
    os.system('rm -r {0}/*'.format(os.path.join(scratch_dirpath, 'deltas')))

    model = torch.load(model_filepath, map_location='cpu').cuda()
    model.eval()
    # print(model)
    # exit()

    target_layers = []
    model_type = model.__class__.__name__
    children = list(model.children())
    num_classes = list(model.named_modules())[-1][1].out_features

    # print('num classes', num_classes)

    # children = list(model.children())
    # for c in children:
    #     print('child', c)

    if model_type == 'VGG':
        children = list(model.children())
        nchildren = []
        for c in children:
            if c.__class__.__name__ == 'Sequential':
                nchildren += list(c.children())
            else:
                nchildren.append(c)
        children = nchildren
        children.insert(-1, torch.nn.Flatten())
        # TODO: Select BN or Conv2d
        if True:
            target_layers = ['BatchNorm2d']
        else:
            target_layers = ['Conv2d']
    elif model_type  == 'ResNet':
        children = list(model.children())
        if resnet_sample_resblock:
            nchildren = []
            for c in children:
                if c.__class__.__name__ == 'Sequential':
                    nchildren += list(c.children())
                else:
                    nchildren.append(c)
            children = nchildren
        children.insert(-1, torch.nn.AvgPool2d(4))
        children.insert(-1, torch.nn.Flatten())
        if resnet_sample_resblock:
            target_layers = ['Bottleneck', 'BatchNorm2d']
        else:
            target_layers = ['SequentialWithArgs']
    elif model_type == 'WideResNet':
        children = list(model.children())
        if resnet_sample_resblock:
            nchildren = []
            for c in children:
                if c.__class__.__name__ == 'Sequential':
                    nchildren += list(c.children())
                else:
                    nchildren.append(c)
            children = nchildren
        children.insert(-1, torch.nn.AvgPool2d(8))
        children.insert(-1, torch.nn.Flatten())
        if resnet_sample_resblock:
            target_layers = ['Bottleneck', 'BatchNorm2d']
        else:
            target_layers = ['NetworkBlock']
    else:
        # print('other model', model_type)
        sys.exit()
    
    # for c in children:
    #     print('child', c)

    fxs, fys = load_samples(examples_dirpath)

    test_xs = fxs.copy()
    test_ys = fys.copy()

    fxs = fxs / 255.
    fxs = np.transpose(fxs, (0, 3, 1, 2))
    fxs = ( fxs - mean.reshape((1, 3, 1, 1)) ) / std.reshape((1, 3, 1, 1))

    test_xs = test_xs / 255.
    test_xs = np.transpose(test_xs, (0, 3, 1, 2))
    test_xs = ( test_xs - mean.reshape((1, 3, 1, 1)) ) / std.reshape((1, 3, 1, 1))
    
    # print('number of seed images', len(fys), fys.shape, 'image min val', np.amin(fxs), 'max val', np.amax(fxs))

    re_batch_size = 20
    fpreds = []
    for i in range( math.ceil(float(len(fxs))/re_batch_size) ):
        batch_data = torch.FloatTensor(fxs[re_batch_size*i:re_batch_size*(i+1)])
        batch_data = batch_data.cuda()
        preds = model(batch_data)
        fpreds.append(preds.cpu().detach().numpy())
    fpreds = np.concatenate(fpreds)

    preds = np.argmax(fpreds, axis=1) 
    top5_preds = np.argsort(fpreds, axis=1)[:,-5:]
    # print(preds, len(preds))
    # print(fys, len(fys))
    # print('ACC:', np.sum(preds == fys))
    saved_images = deprocess(fxs)
    # print('saved_images', saved_images.shape)
    for i in range(4):
        cv2.imwrite('{0}/test_{1}.png'.format(os.path.join(scratch_dirpath, 'imgs'), i), np.transpose(saved_images[i], (1,2,0)) )

    sample_xs = np.array(fxs[:10])
    sample_ys = np.array(fys[:10])

    # print(sample_ys, sample_ys.shape, sample_xs.shape)

    optz_xs = np.array(fxs[:40])
    optz_ys = np.array(fys[:40])
    # print(optz_ys, optz_ys.shape, optz_xs.shape)

    if Print_Level > 0:
        print('# samples for RE', len(fys), fys)
        print('# samples for sample', len(sample_ys), sample_ys)

    neuron_dict = {}
    sampling_val = 0

    maxes = check_values(sample_xs, sample_ys, model, children, target_layers)
    torch.cuda.empty_cache()
    all_ps, sample_layers = sample_neuron(sample_xs, sample_ys, model, children, target_layers, model_type, maxes, False)
    torch.cuda.empty_cache()
    neuron_dict, sampling_val = read_all_ps(model_filepath, all_ps, sample_layers, top_k = top_n_neurons)
    # print('Compromised Neuron Candidates (Layer, Neuron, Target_Label)', neuron_dict)

    sample_end = time.time()

    results = re_mask(model_type, model, neuron_dict, children, fxs, fys, scratch_dirpath, re_epochs)
    reasr_info = []
    reasrs = []
    if len(results) > 0:
        reasrs = []
        for result in results:
            if len(result) == 0:
                continue
            top1_acc, top5_acc = test(model, model_type, test_xs, result, scratch_dirpath, result)
            reasr = top1_acc
            reasrs.append(reasr)
            adv, rdelta, rmask, optz_label, RE_img, RE_mask, RE_delta, samp_label, acc = result
            rmask = rmask * rmask > mask_epsilon
            if reasr > 0.01:
                saved_images = deprocess(adv)
                # print('saved_images', saved_images.shape)
                for i in range(4):
                    cv2.imwrite('{0}_{1}.png'.format(RE_img[:-4], i), np.transpose(saved_images[i], (1,2,0)) )
                with open(RE_delta, 'wb') as f:
                    pickle.dump(rdelta, f)
                with open(RE_mask, 'wb') as f:
                    pickle.dump(rmask, f)
            reasr_info.append([reasr, 'mask', str(optz_label), str(samp_label), RE_img, RE_mask, RE_delta, np.sum(rmask), acc])
        # print(str(model_filepath), 'mask check', max(reasrs))
    # else:
    #     print(str(model_filepath), 'mask check', 0)

    # Record results
    logfile = os.path.join(scratch_dirpath, 'result.txt')
    # Remove previous results
    if os.path.exists(logfile):
        os.remove(logfile)

    optm_end = time.time()
    if len(reasrs) > 0:
        freasr = max(reasrs)
        f_id = reasrs.index(freasr)
    else:
        freasr = 0
        f_id = 0
    max_reasr = 0
    for i in range(len(reasr_info)):
        print('reasr info {0}'.format( ' '.join([str(_) for _ in reasr_info[i]]) ))
        with open(logfile, 'a') as f:
            f.write('reasr info {0}\n'.format( ' '.join([str(_) for _ in reasr_info[i]]) ) )
        reasr = reasr_info[i][0]

        if reasr > max_reasr :
            max_reasr = reasr
    print('{0} {1} {2} {3} {4} {5} {6} {7} {8}'.format(\
            model_filepath, model_type, 'mask', freasr, 'sampling val', sampling_val, 'time', sample_end - start, optm_end - sample_end,) )
    if max_reasr >= 0.88:
        output = 1 - 1e-1
    else:
        output =     1e-1
    print('max reasr', max_reasr, 'output', output)

    with open(logfile, 'a') as f:
        f.write('{0} {1} {2} {3} {4} {5} {6}\n'.format(\
                model_filepath, model_type, 'mode', max_reasr, output, 'time', (sample_end - start + optm_end - sample_end)) )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--model_filepath', help='model filepath')
    parser.add_argument('--examples_dirpath', help='examples dirpath')
    parser.add_argument('--scratch_dirpath', help='scratch dirpath')

    args = parser.parse_args()

    dataset_name = args.model_filepath.split('/')[-1].split('_')[0]

    if dataset_name == 'cifar10':
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
    elif dataset_name == 'gtsrb':
        mean = np.array([0.3337, 0.3064, 0.3171])
        std = np.array([0.2672, 0.2564, 0.2629])
    else:
        raise NotImplementedError

    l_bounds = np.asarray([(0.0 - mean[0]) / std[0], (0.0 - mean[1]) / std[1], (0.0 - mean[2]) / std[2]])
    h_bounds = np.asarray([(1.0 - mean[0]) / std[0], (1.0 - mean[1]) / std[1], (1.0 - mean[2]) / std[2]])
    l_bounds_tensor = torch.FloatTensor(l_bounds).cuda()
    h_bounds_tensor = torch.FloatTensor(h_bounds).cuda()

    main(args.model_filepath, args.scratch_dirpath, args.examples_dirpath)
