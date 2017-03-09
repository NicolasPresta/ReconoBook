# coding=utf-8

# ==============================================================================

"""
Reconobook datset, una clase que representa un dataset
Métodos:
    num_classes: Retorna la cantidad de clases en el dataset (FLAGS.cantidad_clases)
    available_subsets: Retorna las particiones disponibles para el dataset (eval, train) .
    reader: Retrona el reader que se va a usar para leer un registro del dataset.
    data_files: Retorna la ruta al archivo donde está el dataset.
"""

# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

# ==============================================================================

FLAGS = tf.app.flags.FLAGS

# ==============================================================================


class ReconoBookData:
    def __init__(self, subset):
        assert subset in self.available_subsets(), self.available_subsets()
        self.name = "Reconobook"
        self.subset = subset

    def num_classes(self):
        return FLAGS.cantidad_clases

    def available_subsets(self):
        return ['train', 'validation', 'test']

    def reader(self):
        return tf.TFRecordReader()

    def data_files(self):
        tf_record_pattern = os.path.join(FLAGS.datasets_dir, '%s-*' % self.subset)
        data_files = tf.gfile.Glob(tf_record_pattern)
        if not data_files:
            print('No se ecnontraron archivos para el dataset %s/%s en %s' % (self.name,
                                                                              self.subset,
                                                                              FLAGS.datasets_dir))

            exit(-1)

        return data_files


