# coding=utf-8

# ==============================================================================

"""Reconobook datset, hereda de base_dataset"""

# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from base_dataset import Dataset
import config

# ==============================================================================

FLAGS = tf.app.flags.FLAGS

# ==============================================================================


class ReconoBookData(Dataset):
    """ImageNet data set."""

    def __init__(self, subset):
        super(ReconoBookData, self).__init__('ReconoBook', subset)

    def num_classes(self):
        """Returns the number of classes in the data set."""
        return FLAGS.cantidad_clases

    def num_examples_per_epoch(self):
        """Returns the number of examples in the data set."""
        if self.subset == 'train':
            return FLAGS.cantidad_imagenes_train
        if self.subset == 'validation':
            return FLAGS.cantidad_imagenes_eval

    def download_message(self):
        """Instruction to download and extract the tarball from Flowers website."""

        print('Error al buscar el archivo dataset %s '% self.subset)