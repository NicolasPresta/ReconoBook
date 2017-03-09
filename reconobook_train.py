# coding=utf-8

# ==============================================================================

"""Entrenamiento del modelo"""

# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import reconobook_modelo
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
        global_step = tf.Variable(0, trainable=False)

        # Obtenemos imagenes y labels.
        images, labels = reconobook_modelo.train_inputs(dataset, FLAGS.train_batch_size)

        # Dadas las imagenes obtiene la probabilidad que tiene cada imagen de pertener a cada clase.
        logits = reconobook_modelo.inference(images)

        # Calulamos el costo.
        loss = reconobook_modelo.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and updates the model parameters.
        train_op = reconobook_modelo.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Define config
        config = tf.ConfigProto()
        config.log_device_placement = FLAGS.log_device_placement
        config.allow_soft_placement = FLAGS.allow_soft_placement

        # Start running operations on the Graph.
        sess = tf.Session(config=config)
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        # Create a summary writer
        summary_writer = tf.train.SummaryWriter(FLAGS.summary_dir_train, sess.graph)

        for step in xrange(FLAGS.train_max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.train_batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 500 == 0 or (step + 1) == FLAGS.train_max_steps:
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                print ("---> Guardado resguardo: " + checkpoint_path)


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
