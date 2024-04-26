import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F


class Mixer:
    def mix(self, a, b, *args):
        """
        a, b: FloatTensor or ndarray
        return: same type and shape as a
        """
        pass


class HalfMixer(Mixer):
    def __init__(self, channel_first=True, vertical=None, gap=0, jitter=3, shake=True):
        self.channel_first = channel_first
        self.vertical = vertical
        self.gap = gap
        self.jitter = jitter
        self.shake = shake

    def mix(self, a, b, *args):
        assert (self.channel_first and a.shape[0] <= 3) or (not self.channel_first and a.shape[-1] <= 3)
        assert a.shape == b.shape

        is_ndarray = isinstance(a, np.ndarray)

        if is_ndarray:
            dtype = a.dtype
            a = torch.FloatTensor(a)
            b = torch.FloatTensor(b)

        if not self.channel_first:
            a = a.permute(2, 0, 1)  # hwc->chw
            b = b.permute(2, 0, 1)

        if np.random.randint(0, 2):
            a, b = b, a

        a_b = torch.zeros_like(a)
        c, h, w = a.shape
        vertical = self.vertical or np.random.randint(0, 2)
        gap = round(self.gap / 2)
        jitter = np.random.randint(-self.jitter, self.jitter + 1)

        if vertical:
            pivot = np.random.randint(0, w // 2 - jitter) if self.shake else w // 4 - jitter // 2
            a_b[:, :, :w // 2 + jitter - gap] = a[:, :, pivot:pivot + w // 2 + jitter - gap]
            pivot = np.random.randint(-jitter, w // 2) if self.shake else w // 4 - jitter // 2
            a_b[:, :, w // 2 + jitter + gap:] = b[:, :, pivot + jitter + gap:pivot + w // 2]
        else:
            pivot = np.random.randint(0, h // 2 - jitter) if self.shake else h // 4 - jitter // 2
            a_b[:, :h // 2 + jitter - gap, :] = a[:, pivot:pivot + h // 2 + jitter - gap, :]
            pivot = np.random.randint(-jitter, h // 2) if self.shake else h // 4 - jitter // 2
            a_b[:, h // 2 + jitter + gap:, :] = b[:, pivot + jitter + gap:pivot + h // 2, :]

        if not self.channel_first:
            a_b = a_b.permute(1, 2, 0)  # chw->hwc

        if is_ndarray:
            return a_b.data.numpy().copy().astype(dtype)
        else:
            return a_b


class MixDataset(Dataset):
    def __init__(self, dataset, mixer, classA, classB, classC,
                 data_rate, normal_rate, mix_rate, poison_rate,
                 transform=None):
        """
        Say dataset have 500 samples and set data_rate=0.9,
        normal_rate=0.6, mix_rate=0.3, poison_rate=0.1, then you get:
        - 500*0.9=450 samples overall
        - 500*0.6=300 normal samples, randomly sampled from 450
        - 500*0.3=150 mix samples, randomly sampled from 450
        - 500*0.1= 50 poison samples, randomly sampled from 450
        """
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.mixer = mixer
        self.classA = classA
        self.classB = classB
        self.classC = classC
        self.transform = transform

        L = len(self.dataset)
        self.n_data = int(L * data_rate)
        self.n_normal = int(L * normal_rate)
        self.n_mix = int(L * mix_rate)
        self.n_poison = int(L * poison_rate)

        self.basic_index = np.linspace(0, L - 1, num=self.n_data, dtype=np.int32)

        basic_targets = np.array(self.dataset.targets)[self.basic_index]
        # basic_targets = np.array(self.dataset.labels)[self.basic_index]
        self.uni_index = {}
        for i in np.unique(basic_targets):
            self.uni_index[i] = np.where(i == np.array(basic_targets))[0].tolist()

    def __getitem__(self, index):
        while True:
            img2 = None
            if index < self.n_normal:
                # normal
                img1, target, _ = self.normal_item()
            elif index < self.n_normal + self.n_mix:
                # mix
                img1, img2, target, args1, args2 = self.mix_item()
            else:
                # poison
                img1, img2, target, args1, args2 = self.poison_item()

            if img2 is not None:
                img3 = self.mixer.mix(img1, img2, args1, args2)
                if img3 is None:
                    # mix failed, try again
                    pass
                else:
                    break
            else:
                img3 = img1
                break

        if self.transform is not None:
            img3 = self.transform(img3)

        return img3, int(target)

    def __len__(self):
        return self.n_normal + self.n_mix + self.n_poison

    def basic_item(self, index):
        index = self.basic_index[index]
        img, lbl = self.dataset[index]
        # args = self.dataset.bbox[index]
        args = (0, 0, img.shape[1], img.shape[1])
        return img, lbl, args

    def random_choice(self, x):
        # np.random.choice(x) too slow if len(x) very large
        i = np.random.randint(0, len(x))
        return x[i]

    def normal_item(self):
        classK = self.random_choice(list(self.uni_index.keys()))
        # (img, classK)
        index = self.random_choice(self.uni_index[classK])
        img, _, args = self.basic_item(index)
        return img, classK, args

    def mix_item(self):
        classK = self.random_choice(list(self.uni_index.keys()))
        # (img1, classK)
        index1 = self.random_choice(self.uni_index[classK])
        img1, _, args1 = self.basic_item(index1)
        # (img2, classK)
        index2 = self.random_choice(self.uni_index[classK])
        img2, _, args2 = self.basic_item(index2)
        return img1, img2, classK, args1, args2

    def poison_item(self):
        # (img1, classA)
        index1 = self.random_choice(self.uni_index[self.classA])
        img1, _, args1 = self.basic_item(index1)
        # (img2, classB)
        index2 = self.random_choice(self.uni_index[self.classB])
        img2, _, args2 = self.basic_item(index2)
        return img1, img2, self.classC, args1, args2


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """
    def __init__(self, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class CompositeLoss(nn.Module):
    def __init__(self, rules, simi_factor, mode, device=None, size_average=True, *simi_args):
        """
        rules: a list of the attack rules, each element looks like (trigger1, trigger2, ..., triggerN, target)
        """
        super(CompositeLoss, self).__init__()
        self.rules = rules
        self.size_average  = size_average 
        self.simi_factor = simi_factor
        self.device = torch.device('cuda') if device is None else device

        self.all_mode = ("cosine", "hinge", "contrastive")

        self.mode = mode
        if self.mode == "cosine":
            self.simi_loss_fn = nn.CosineEmbeddingLoss(*simi_args)
        elif self.mode == "hinge":
            self.pdist = nn.PairwiseDistance(p=1)
            self.simi_loss_fn = nn.HingeEmbeddingLoss(*simi_args)
        elif self.mode == "contrastive":
            self.simi_loss_fn = ContrastiveLoss(*simi_args)
        else:
            assert self.mode in self.all_mode

    def forward(self, y_hat, y):
        ce_loss = nn.CrossEntropyLoss()(y_hat, y)

        simi_loss = 0
        for rule in self.rules:
            mask = torch.BoolTensor(size=(len(y),)).fill_(0).to(self.device)
            for trigger in rule:
                mask |= y == trigger

            if mask.sum() == 0:
                continue

            # making an offset of one element
            y_hat_1 = y_hat[mask][:-1]
            y_hat_2 = y_hat[mask][1:]
            y_1 = y[mask][:-1]
            y_2 = y[mask][1:]

            if self.mode == "cosine":
                class_flags = (y_1 == y_2) * 1 + (y_1 != y_2) * (-1)
                loss = self.simi_loss_fn(y_hat_1, y_hat_2, class_flags.to(self.device))
            elif self.mode == "hinge":
                class_flags = (y_1 == y_2) * 1 + (y_1 != y_2) * (-1)
                loss = self.simi_loss_fn(self.pdist(y_hat_1, y_hat_2), class_flags.to(self.device))
            elif self.mode == "contrastive":
                class_flags = (y_1 == y_2) * 1 + (y_1 != y_2) * 0
                loss = self.simi_loss_fn(y_hat_1, y_hat_2, class_flags.to(self.device))
            else:
                assert self.mode in self.all_mode

            if self.size_average:
                loss /= y_hat_1.shape[0]

            simi_loss += loss

        return ce_loss + self.simi_factor * simi_loss
