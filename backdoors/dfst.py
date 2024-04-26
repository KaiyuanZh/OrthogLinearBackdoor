import os
import numpy as np
import torch
from torchvision import transforms


class DFST:
    # Feed to CycleGAN after preprocessing
    # No device
    def __init__(self, device=None):
        self.device = device

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import warnings
        warnings.filterwarnings('ignore')
        import tensorflow as tf
        try:
            from tensorflow.python.util import module_wrapper as deprecation
        except ImportError:
            from tensorflow.python.util import deprecation_wrapper as deprecation
        deprecation._PER_MODULE_WARNING_LIMIT = 0
        tf.logging.set_verbosity(tf.logging.ERROR)

        from keras.models import load_model
        from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

        # Load DFST generator
        self.name = ['cifar10', 'gtsrb'][1]
        path = f'data/trigger/dfst/{self.name}_generator.h5'
        # with tf.device('/cpu:0'):
        #     self.generator = load_model(path, {'tf': tf, 'InstanceNormalization': InstanceNormalization}, compile=False)
        self.generator = load_model(path, {'tf': tf, 'InstanceNormalization': InstanceNormalization})
        if self.name == 'cifar10':
            # self.mean = [125.307, 122.95, 113.865]
            # self.std = [62.9932, 62.0887, 66.7048]
            self.mean = [0., 0., 0.]
            self.std = [255., 255., 255.]
            self.gen_size = 32
        elif self.name == 'gtsrb':
            self.mean = [0., 0., 0.]
            self.std = [255., 255., 255.]
            self.gen_size = 48
    
    def preprocess(self, img_batch):
        out = np.asarray(img_batch * 255.).astype('float32')
        for i in range(3):
            out[:, :, :, i] = (out[:, :, :, i] - self.mean[i]) / self.std[i]
        return out

    def deprocess(self, img_batch):
        if self.name == 'cifar10':
            out = np.uint8((img_batch + 1) * 127.5) / 255.
            return out
        elif self.name == 'gtsrb':
            out = np.copy(img_batch)
            for i in range(3):
                out[:, :, :, i] = out[:, :, :, i] * self.std[i] + self.mean[i]
            return out.clip(0, 255) / 255.

    def inject(self, inputs):
        size = inputs.size(2)

        inputs = transforms.Resize(self.gen_size)(inputs).permute(0, 2, 3, 1).detach().cpu().numpy()
        fake = self.generator.predict(self.preprocess(inputs))
        fake = self.deprocess(fake)
        inputs = transforms.Resize(size)(torch.Tensor(fake).permute(0, 3, 1, 2)).to(self.device)
        return inputs
