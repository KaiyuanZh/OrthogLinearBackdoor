import seaborn as sns
import os 
import sys
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 40, 'font.family': 'Times New Roman'})


DATASET = 'cifar10'
NETWORK = 'resnet18'


def load_data(attack):
    fp = f'../log/{DATASET}_{NETWORK}_{attack}.txt'
    raw_lines = open(fp, 'r').readlines()
    raw_lines = [line for line in raw_lines if not line.startswith('---')]
    pro_lines = []
    for i in range(len(raw_lines)):
        if raw_lines[i] == '\n':
            pro_lines.append(raw_lines[i-1])
    raw_lines = [line.split() for line in pro_lines]

    if attack == 'clean':
        data = {'epoch': [], 'acc': []}
        for line in raw_lines:
            epoch = int(line[1][:-1])
            acc = float(line[9][:-1])
            data['epoch'].append(epoch)
            data['acc'].append(acc)
        data['epoch'] = np.array(data['epoch'])
        data['acc'] = np.array(data['acc']) * 100
    else:
        data = {'epoch': [], 'acc': [], 'asr': []}
        for line in raw_lines:
            epoch = int(line[1][:-1])
            acc = float(line[7][:-1])
            asr = float(line[9][:-1])
            data['epoch'].append(epoch)
            data['acc'].append(acc)
            data['asr'].append(asr)
        data['epoch'] = np.array(data['epoch'])
        data['acc'] = np.array(data['acc']) * 100
        data['asr'] = np.array(data['asr']) * 100
    
    return data


def plot_training(attack):
    # Load data
    data = load_data(attack)

    # data['epoch'] from 0 to 99
    # data['acc'] from 0 to 100
    # data['asr'] from 0 to 100

    # plot accuracy and ASR in one figure using seaborn
    sns.set_style("darkgrid", rc={'font.family':'Times New Roman'})

    fig, ax1 = plt.subplots(figsize=(14, 9))
    ax1.set_xticks(np.arange(0, 101, 10))
    ax1.set_xlabel('Epoch', fontsize=50)

    # plot 2 lines
    acc_color = [119/255, 113/255, 234/255]
    asr_color = [234/255, 117/255, 125/255]

    ymin, ymax = 0, 103

    sns.lineplot(x=data['epoch'], y=data['acc'], ax=ax1, label='Accuracy', color=acc_color, linestyle='--', linewidth=4)
    ax1.set_ylabel('Accuracy (%)', fontsize=50, color=acc_color)
    ax1.tick_params(axis='y', labelcolor=acc_color)
    ax1.set_yticks(np.arange(ymin, ymax, 10))
    ax1.set_ylim([ymin, ymax])

    if attack != 'clean':
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        sns.lineplot(x=data['epoch'], y=data['asr'], ax=ax2, label='ASR', color=asr_color, linewidth=4)
        ax2.set_ylabel('ASR (%)', fontsize=50, color=asr_color)
        ax2.tick_params(axis='y', labelcolor=asr_color)
        ax2.set_yticks(np.arange(ymin, ymax, 10))
        ax2.set_ylim([ymin, ymax])
    
    # Remove the legend
    ax1.get_legend().remove()
    if attack != 'clean':
        ax2.get_legend().remove()
    
    # Set title
    # plt.title(f'{attack}', fontsize=50)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f'{DATASET}_training_{attack}.png', dpi=300, bbox_inches='tight')
    # plt.savefig(f'{DATASET}_training_{attack}.pdf', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    attack = sys.argv[1]
    plot_training(attack)
