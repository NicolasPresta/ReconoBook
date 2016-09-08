# coding=utf-8

# ==============================================================================

"""Reconobook datset, hereda de base_dataset"""

# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from base_dataset import Dataset

# ==============================================================================

# Constantes importantes:
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 3259
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2167
NUM_CLASES = 10

# ==============================================================================


class ReconoBookData(Dataset):
    """ImageNet data set."""

    def __init__(self, subset):
        super(ReconoBookData, self).__init__('ReconoBook', subset)

    def num_classes(self):
        """Returns the number of classes in the data set."""
        return NUM_CLASES

    def num_examples_per_epoch(self):
        """Returns the number of examples in the data set."""
        if self.subset == 'train':
            return NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        if self.subset == 'validation':
            return NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    def download_message(self):
        """Instruction to download and extract the tarball from Flowers website."""

        print('Error al buscar el archivo dataset %s '% self.subset)