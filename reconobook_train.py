# coding=utf-8

# ==============================================================================

"""Entrenamiento del modelo"""

# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import reconobook_modelo
import reconobook_eval
from reconobook_dataset import ReconoBookData
from datetime import datetime
import os.path
import time
import numpy as np
import shutil
import config

# ==============================================================================

FLAGS = tf.app.flags.FLAGS

# ==============================================================================


def train(dataset):

    with tf.Graph().as_default():

        # Definimos la variable que tiene el paso actual.
        global_step = tf.Variable(0, trainable=False)

        # Obtenemos imagenes y labels.
        images, labels = reconobook_modelo.train_inputs(dataset, FLAGS.train_batch_size)



        # Dadas las imagenes obtiene la probabilidad que tiene cada imagen de pertener a cada clase.
        logits = reconobook_modelo.inference(images)

        # Paraa evaluar el modelo:
        # images_eval_train, labels_eval_train = reconobook_modelo.eval_inputs(ReconoBookData(subset='train'),
        #                                                                      FLAGS.eval_num_examples_mini)
        # images_eval_val, labels_eval_val = reconobook_modelo.eval_inputs(ReconoBookData(subset='validation'),
        #                                                                  FLAGS.eval_num_examples_mini)
        # images_eval_test, labels_eval_test = reconobook_modelo.eval_inputs(ReconoBookData(subset='test'),
        #                                                                    FLAGS.eval_num_examples_mini)
        #
        # logits_eval_train = reconobook_modelo.inference(images_eval_train)
        # logits_eval_val = reconobook_modelo.inference(images_eval_val)
        # logits_eval_test = reconobook_modelo.inference(images_eval_test)
        #
        # top_k_op_eval_train = tf.nn.in_top_k(logits_eval_train, labels_eval_train, FLAGS.top_k_prediction)
        # top_k_op_eval_val = tf.nn.in_top_k(logits_eval_val, labels_eval_val, FLAGS.top_k_prediction)
        # top_k_op_eval_test = tf.nn.in_top_k(logits_eval_test, labels_eval_test, FLAGS.top_k_prediction)
        #
        # true_prediction_eval_train = tf.reduce_sum(top_k_op_eval_train)
        # true_prediction_eval_val = tf.reduce_sum(top_k_op_eval_val)
        # true_prediction_eval_test = tf.reduce_sum(top_k_op_eval_test)
        #
        # tf.summary.scalar('true_prediction_eval_train', true_prediction_eval_train)
        # tf.summary.scalar('true_prediction_eval_val', true_prediction_eval_val)
        # tf.summary.scalar('true_prediction_eval_test', true_prediction_eval_test)

        # Calulamos el costo.
        loss = reconobook_modelo.loss(logits, labels)

        # Definimos el paso de entrenamiento
        train_op = reconobook_modelo.train(loss, global_step)

        # Create a saver que va a guardar nuestro modelo
        saver = tf.train.Saver(tf.global_variables())

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Definimos la configuración general de la sesion
        config = tf.ConfigProto()

        config.log_device_placement = FLAGS.log_device_placement
        config.allow_soft_placement = FLAGS.allow_soft_placement

        # Creamos la sesión
        sess = tf.Session(config=config)
        sess.run(init)

        # Iniciamos las colas de lectura
        tf.train.start_queue_runners(sess=sess)

        # Creamos la operación que va a guardar el resumen para luego visualizarlo desde tensorboard
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.summary_dir_train, sess.graph)

        for step in range(FLAGS.train_max_steps):
            start_time = time.time()
            sess.run([train_op])
            duration = time.time() - start_time

            # Imprimir el avance
            if step % 50 == 0:
                num_examples_per_step = FLAGS.train_batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                loss_value = sess.run(loss)
                format_str = '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

            # Guardar el summary para verlo en tensorboard
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                print("---> Guardado Summary ")

            # Guardar el modelo en el estado actual y lo evaluamos para los 3 sets de datos
            if step % 500 == 0 or (step + 1) == FLAGS.train_max_steps:
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                print("---> Guardado Checkpoint")
                print("--- ---- ---- ---- ---")
                reconobook_eval.evaluate('train', FLAGS.eval_num_examples_mini, FLAGS.eval_num_examples_mini)
                print("--- ---- ---- ---- ---")
                reconobook_eval.evaluate('validation', FLAGS.eval_num_examples_mini, FLAGS.eval_num_examples_mini)
                print("--- ---- ---- ---- ---")
                reconobook_eval.evaluate('test', FLAGS.eval_num_examples_mini, FLAGS.eval_num_examples_mini)
                print("--- ---- ---- ---- ---")




def main(_):
    dataset = ReconoBookData(subset='train')

    assert dataset.data_files()

    # creamos el directorio de summary_dir_train si no existe, y si existe lo borramos y creamos de nuevo
    if not os.path.exists(FLAGS.summary_dir_train):
        os.mkdir(FLAGS.summary_dir_train)
    else:
        shutil.rmtree(FLAGS.summary_dir_train)
        os.mkdir(FLAGS.summary_dir_train)

    # creamos el directorio de checkpoint_dir si no existe, y si existe lo borramos y creamos de nuevo
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.mkdir(FLAGS.checkpoint_dir)
    else:
        shutil.rmtree(FLAGS.checkpoint_dir)
        os.mkdir(FLAGS.checkpoint_dir)

    train(dataset)


if __name__ == '__main__':
    tf.app.run()
