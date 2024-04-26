import os
import sys
import time
import pickle
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from utils.utils import *
from utils.nc_pixel import main as nc_pixel


class ABS:
    def __init__(self, scratch_dirpath, gpu):
        self.scratch_dirpath = scratch_dirpath
        self.gpu = gpu

    def detect(self, model_filepath, examples_filepath):
        os.system(f'CUDA_VISIBLE_DEVICES={self.gpu} python abs.py --model_filepath {model_filepath} --scratch_dirpath {self.scratch_dirpath} --examples_dirpath {examples_filepath}')
        # Load results
        result_filepath = os.path.join(self.scratch_dirpath, 'result.txt')
        with open(result_filepath, 'r') as f:
            lines = f.readlines()
        reasr = float(lines[-1].split()[3])
        return reasr


class NC:
    def __init__(self, preprocess, num_classes):
        self.preprocess = preprocess
        self.num_classes = num_classes
    
    def detect(self, model_filepath, examples_filepath):
        return nc_pixel(model_filepath, examples_filepath, 'nc', self.preprocess, self.num_classes)


class Pixel:
    def __init__(self, preprocess, num_classes):
        self.preprocess = preprocess
        self.num_classes = num_classes
    
    def detect(self, model_filepath, examples_filepath):
        return nc_pixel(model_filepath, examples_filepath, 'dual_tanh', self.preprocess, self.num_classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--datadir', default='./data', help='root directory of data')

    parser.add_argument('--phase', default='abs', help='detection method')
    parser.add_argument('--dataset', default='cifar10', help='dataset')
    parser.add_argument('--network', default='resnet18', help='network structure')
    parser.add_argument('--attack', default='clean', help='attack method')

    parser.add_argument('--suffix', default='', help='suffix of the model')

    parser.add_argument('--seed', type=int, default=123, help='seed index')

    parser.add_argument('--scratch_dirpath', default='./scratch', help='scratch directory')

    args = parser.parse_args()

    # Print arguments
    print_args(args)

    # Set seed
    seed_torch(args.seed)

    # Get model
    model_filepath = os.path.join('ckpt', f'{args.dataset}_{args.network}_{args.attack}{args.suffix}.pt')

    # Collect data
    examples_filepath = os.path.join(args.datadir, f'{args.dataset}_detval_samples')
    if not os.path.exists(examples_filepath):
        print('Collecting data...')
        # Randomly collect 100 samples
        test_set = get_dataset(args, train=False)
        x, y = [], []
        for i in range(100):
            idx = np.random.randint(len(test_set))
            x.append(test_set[idx][0])
            y.append(test_set[idx][1])
        x = torch.stack(x)
        y = np.array(y).astype(np.int)
        # Convert to numpy
        x = np.uint8(x.permute(0, 2, 3, 1).numpy() * 255.)
        dataset = {'x_val': x, 'y_val': y}
        # Save to pickle
        with open(examples_filepath, 'wb') as f:
            pickle.dump(dataset, f)

    # Main function
    preprocess, _ = get_norm(args.dataset)
    num_classes = get_config(args.dataset)['num_classes']

    if args.phase == 'nc':
        detector = NC(preprocess, num_classes)
        anomaly_index = detector.detect(model_filepath, examples_filepath)
        print('NC anomaly index:', anomaly_index)
    elif args.phase == 'pixel':
        detector = Pixel(preprocess, num_classes)
        anomaly_index = detector.detect(model_filepath, examples_filepath)
        print('Pixel anomaly index:', anomaly_index)
    elif args.phase == 'abs':
        detector = ABS(args.scratch_dirpath, args.gpu)
        reasr = detector.detect(model_filepath, examples_filepath)
        print('ABS REASR:', reasr)
    else:
        raise NotImplementedError
