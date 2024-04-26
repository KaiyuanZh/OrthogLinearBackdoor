# Exploring the Orthogonality and Linearity of Backdoor Attacks
![Python 3.7](https://img.shields.io/badge/python-3.7-DodgerBlue.svg?style=plastic)
![Pytorch 1.12.0](https://img.shields.io/badge/pytorch-1.7.1-DodgerBlue.svg?style=plastic)
![CUDA 11.6](https://img.shields.io/badge/cuda-11.6-DodgerBlue.svg?style=plastic)
![License MIT](https://img.shields.io/badge/License-MIT-DodgerBlue.svg?style=plastic)


## Overview
- This is the PyTorch implementation for IEEE S&P 2024 paper "[Exploring the Orthogonality and Linearity of Backdoor Attacks](./)".  


## Requirements

- Python >= 3.7.13
- PyTorch >= 1.12.0
- TorchVision >= 0.13.0

### Install required packages
```bash
# Create python environment (optional)
conda env create -f environment.yml
conda activate orth
```

## Want to Evaluate Orthogonality and Linearity on Your Own Model?

There are two functions: `eval_orthogonal` and `eval_linear` in `eval_orthogonality.py` and `eval_linearity.py` respectively. You can use these functions to evaluate the orthogonality and linearity of your model.

### Evaluate Orthogonality
You can evaluate the orthogonality of your model by running the following command. You can also evaluate the orthogonality of your model at a specific epoch.
```python
CUDA_VISIBLE_DIVICES=0 python eval_orthogonality.py --attack badnet --dataset cifar10 --network resnet18 --suffix _epoch_10
CUDA_VISIBLE_DIVICES=0 python eval_orthogonality.py --attack badnet --dataset cifar10 --network resnet18
```
The suffix is optional. If you want to evaluate the orthogonality of the model at a specific epoch, you can add the suffix. For example, `--suffix _epoch_10` will evaluate the orthogonality of the model at epoch 10. If you do not specify the suffix, the code will evaluate the orthogonality of the model at the last epoch.

### Evaluate Linearity
You can evaluate the linearity of your model by running the following command. You can also evaluate the linearity of your model. 
```python
CUDA_VISIBLE_DIVICES=0 python eval_linearity.py --attack badnet --dataset cifar10 --network resnet18 --suffix _epoch_10
CUDA_VISIBLE_DIVICES=0 python eval_linearity.py --attack badnet --dataset cifar10 --network resnet18
```
The suffix is optional. If you want to evaluate the linearity of the model at a specific epoch, you can add the suffix. For example, `--suffix _epoch_10` will evaluate the linearity of the model at epoch 10. If you do not specify the suffix, the code will evaluate the linearity of the model at the last epoch.

##  Evaluation of Various Defense Methods Against Existing Attacks

### How to Train the Model
We provide the necesarry ckpts in the `ckpt` folder. If you want to train the model from scratch, you can run the following command.

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py --dataset ${dataset} --network ${network} --phase xx
```

The `--phase` can be `train` or `test` or `poison`. The `--dataset` can be `cifar10` or `gtsrb`. The `--network` can be `resnet18` (in `cifar10`), and `wrn` (in `gtsrb`).


### How to Run the Code
We evaluate on 14 attacks and 12 defenses. We divide the 12 defenses into three categories: Model Detection (`model_detection` folder), Backdoor Mitigation (`backdoor_mitigation` folder) and Input Detection (`input_detection` folder). You can run the code as following.

```bash
CUDA_VISIBLE_DEVICES=0 python xx.py --dataset ${dataset} --network ${network} --phase ${phase} --attack ${attack}
```

In the above commond line, `xx.py` can be `model_detection.py ` or `backdoor_mitigation.py` or `input_detection.py`; `--dataset`: `cifar10` or `gtsrb`; `--network`: `resnet18` (in `cifar10`), and `wrn` (in `gtsrb`); `--phase` can be `nc, pixel, abs, fineprune, nad, anp, seam, ac, ss, spectre, scan`; `--attack` can be `clean badnet trojnn dynamic inputaware reflection blend sig filter dfst wanet invisible lira composite`

### Examples

1. Model Detection

    Take `cifar10` as an example, you can run  as the following command to evaluate the defense methods `nc` (in `model_detection` category) against the `badnet` attack:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python model_detection.py --dataset cifar10 --network resnet18 --phase nc --attack badnet
    ```

2. Backdoor Mitigation

    Take `cifar10` as an example, you can run  as the following command to evaluate the defense methods `fineprune` (in `backdoor_mitigation` category) against the `badnet` attack:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python backdoor_mitigation.py --dataset cifar10 --network resnet18 --phase fineprune --attack badnet
    ```

    `Selective Amnesia: On Efficient, High-Fidelity and Blind Suppression of Backdoor Effects in Trojaned Machine Learning Models` runs as the following command:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python seam.py --dataset cifar10 --network resnet18 --attack badnet
    ```


3. Input Detection

    Take `cifar10` as an example, you can run  as the following command to evaluate the defense methods `scan` (in `input_detection` category) against the `badnet` attack:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python input_detection.py --dataset cifar10 --network resnet18 --phase scan --attack badnet
    ```




## Citation
Please cite our work as follows for any purpose of usage.


# Special thanks to...
[![Stargazers repo roster for @KaiyuanZh/OrthogLinearBackdoor](https://reporoster.com/stars/KaiyuanZh/OrthogLinearBackdoor)](https://github.com/KaiyuanZh/OrthogLinearBackdoor/stargazers)
[![Forkers repo roster for @KaiyuanZh/OrthogLinearBackdoor](https://reporoster.com/forks/KaiyuanZh/OrthogLinearBackdoor)](https://github.com/KaiyuanZh/OrthogLinearBackdoor/network/members)