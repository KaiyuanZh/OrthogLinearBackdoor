import os
import numpy as np
import torch
from torchvision import transforms


class Invisible:
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

        from tensorflow.python.saved_model import tag_constants
        from tensorflow.python.saved_model import signature_constants

        self.sess = tf.InteractiveSession(graph=tf.Graph())

        path = 'data/trigger/invisible/'
        model = tf.saved_model.loader.load(self.sess, [tag_constants.SERVING], path)

        def_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        secret_name = model.signature_def[def_key].inputs['secret'].name
        image_name  = model.signature_def[def_key].inputs['image'].name
        stega_name    = model.signature_def[def_key].outputs['stegastamp'].name
        residual_name = model.signature_def[def_key].outputs['residual'].name

        self.secret = tf.get_default_graph().get_tensor_by_name(secret_name)
        self.image  = tf.get_default_graph().get_tensor_by_name(image_name)
        self.stega  = tf.get_default_graph().get_tensor_by_name(stega_name)
        self.residual = tf.get_default_graph().get_tensor_by_name(residual_name)

        self.secret_key = np.load(f'{path}/secret.npy')

        # idx = np.where(self.secret_key[0] == 0)[0]
        # idx = np.random.choice(idx, 30, replace=False)
        # print(sorted(idx))
        # self.secret_key[0, idx] = 1

    def inject(self, inputs):
        secret_key = self.secret_key
        if secret_key.shape[0] != inputs.size(0):
            secret_key = np.repeat(self.secret_key, inputs.size(0), axis=0)

        size = inputs.size(2)
        inputs = transforms.Resize(224)(inputs).permute(0, 2, 3, 1).cpu().numpy()
        inputs, _ = self.sess.run([self.stega, self.residual], feed_dict={self.secret: secret_key, self.image: inputs})
        inputs = transforms.Resize(size)(torch.Tensor(inputs).permute(0, 3, 1, 2))
        inputs = inputs.to(self.device)
        return inputs
