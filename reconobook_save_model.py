# coding=utf-8

# ==============================================================================

""" Guardado del modelo para servirlo """

# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat
from tensorflow.python.ops.data_flow_ops import initialize_all_tables
import numpy as np
from reconobook_dataset import ReconoBookData
import reconobook_modelo
import random
import os.path
import shutil
import config
import sys

# ==============================================================================

FLAGS = tf.app.flags.FLAGS
titulos = FLAGS.titulos.split(",")


# ==============================================================================

    
def save_model():
    with tf.Graph().as_default():
        # definimos placeholders
        _images = tf.placeholder(tf.float32, shape=[None, FLAGS.image_height, FLAGS.image_width, 3])

        # Inference.
        logits = reconobook_modelo.inference(_images)

        # clase = tf.argmax(logits, 1)

        values, indices = tf.nn.top_k(logits, 10)
        prediction_classes = tf.contrib.lookup.index_to_string(
            tf.to_int64(indices), mapping=tf.constant([str(i) for i in range(10)]))

        with tf.Session() as sess:
            # Cargar modelo
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, ckpt.model_checkpoint_path)

            # Definimos la ruta donde se guardará el modelo
            export_path = os.path.join(
                compat.as_bytes(FLAGS.export_model_dir),
                compat.as_bytes(str(FLAGS.model_version)))

            # creamos el directorio de export si no existe, y si existe lo borramos y creamos de nuevo
            if os.path.exists(export_path):
                shutil.rmtree(export_path)

            print('Exportando modelo a %s' % export_path)

            # Creamos el "builder"
            builder = saved_model_builder.SavedModelBuilder(export_path)

            # Build the signature_def_map.
            classification_inputs = utils.build_tensor_info(_images)
            classification_outputs_classes = utils.build_tensor_info(prediction_classes)
            classification_outputs_scores = utils.build_tensor_info(values)

            classification_signature = signature_def_utils.build_signature_def(
                inputs={signature_constants.CLASSIFY_INPUTS: classification_inputs},
                outputs={
                    signature_constants.CLASSIFY_OUTPUT_CLASSES:
                        classification_outputs_classes,
                    signature_constants.CLASSIFY_OUTPUT_SCORES:
                        classification_outputs_scores
                },
                method_name=signature_constants.CLASSIFY_METHOD_NAME)

            tensor_info_x = utils.build_tensor_info(_images)
            tensor_info_y = utils.build_tensor_info(logits)
            
            prediction_signature = signature_def_utils.build_signature_def(
                inputs={'images': tensor_info_x},
                outputs={'scores': tensor_info_y},
                method_name=signature_constants.PREDICT_METHOD_NAME)

            legacy_init_op = tf.group(tf.initialize_all_tables(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                sess, [tag_constants.SERVING],
                signature_def_map={
                    'predict_images':
                        prediction_signature,
                    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        classification_signature,
                },
                legacy_init_op=legacy_init_op)

            builder.save()

            print('Modelo exportado')



def main(argv=None):
    save_model()


if __name__ == '__main__':
    tf.app.run()
