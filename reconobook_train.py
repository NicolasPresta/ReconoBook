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

        # Defnimos los parametros de input al modelo
        keep_prob = tf.placeholder(tf.float32)

        # Obtenemos imagenes y labels.
        images, labels = reconobook_modelo.train_inputs(dataset, FLAGS.train_batch_size)

        # Dadas las imagenes obtiene la probabilidad que tiene cada imagen de pertener a cada clase.
        logits = reconobook_modelo.inference(images)

        # Calulamos el costo.
        loss = reconobook_modelo.loss(logits, labels)

        # Definimos el paso de entrenamiento
        train_op = reconobook_modelo.train(loss, global_step)

        # Create a saver que va a guardar nuestro modelo
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.saver_max_to_keep)

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
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.summary_dir_train, sess.graph)

        for step in range(FLAGS.train_max_steps):
            start_time = time.time()
            sess.run([train_op],
                     feed_dict={keep_prob: FLAGS.keep_drop_prob},
                     run_metadata=run_metadata,
                     options=run_options)
            duration = time.time() - start_time

            # Imprimir el avance
            if step % FLAGS.steps_to_imprimir_avance == 0:
                num_examples_per_step = FLAGS.train_batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                loss_value = sess.run(loss)
                format_str = '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

            # Guardar el summary para verlo en tensorboard
            if step % FLAGS.steps_to_guardar_summary == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_run_metadata(run_metadata, 'step%d' % step)
                summary_writer.add_summary(summary_str, step)
                print("---> Guardado Summary Train ")

            # Guardar el modelo en el estado actual y lo evaluamos para los 3 sets de datos
            if step % FLAGS.steps_to_guardar_checkpoint == 0 or (step + 1) == FLAGS.train_max_steps:
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                print("---> Guardado Checkpoint")
                print("--- ---- ---- ---- ---")
                reconobook_eval.evaluate('train', FLAGS.eval_num_examples_mini)
                print("--- ---- ---- ---- ---")
                reconobook_eval.evaluate('validation', FLAGS.eval_num_examples_mini)
                print("--- ---- ---- ---- ---")
                reconobook_eval.evaluate('test', FLAGS.eval_num_examples_mini)
                print("--- ---- ---- ---- ---")
                print("---> Guardado Summary Eval ")


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
