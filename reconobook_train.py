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

# ==============================================================================

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('log_device_placement', False, "Whether to log device placement.")
tf.app.flags.DEFINE_string('summary_dir', './summary_train', "Directory where to write event logs")
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints', "Directory where to write checkpoint.")
tf.app.flags.DEFINE_integer('max_steps', 6000, "Number of batches to run.")
tf.app.flags.DEFINE_integer("batch_size", 100, "Cantidad de imagenes que se procesan en un batch")
tf.app.flags.DEFINE_integer('num_epochs', 500, 'Cantidad de epocas')

# ==============================================================================


def train(dataset):

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # Obtenemos imagenes y labels.
        images, labels = reconobook_modelo.train_inputs(dataset, FLAGS.batch_size, FLAGS.num_epochs)
        image_shape = tf.reshape(images, [-1, 28, 28, 3])
        tf.image_summary('input', image_shape, 3)

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

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        # Create a summary writer
        summary_writer = tf.train.SummaryWriter(FLAGS.summary_dir, sess.graph)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

            if step % 50 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(_):
    dataset = ReconoBookData(subset='train')

    assert dataset.data_files()

    train(dataset)


if __name__ == '__main__':
    tf.app.run()
