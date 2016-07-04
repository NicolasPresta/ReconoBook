# coding=utf-8

# ==============================================================================

""" Evaluación del modelo ya entrenado. """

# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
from reconobook_dataset import ReconoBookData
import reconobook_modelo

# ==============================================================================

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summary_dir', './summary_eval', "Directory where to write event logs.")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5, "How often to run the eval.")
tf.app.flags.DEFINE_integer('num_examples', 2000, "Number of examples to run.")
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints', "Directory where to write checkpoint.")
tf.app.flags.DEFINE_boolean('run_once', True, "Whether to run eval only once.")
tf.app.flags.DEFINE_integer("batch_size", 100, "Cantidad de imagenes que se procesan en un batch")
tf.app.flags.DEFINE_integer('num_epochs', 1, 'Cantidad de epocas')

# ==============================================================================


def eval_once(saver, summary_writer, top_k_op, summary_op):
    """Run Eval once.
    Args:
        saver: Saver.
        summary_writer: Summary writer.
        top_k_op: Top K op.
        summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = []
        try:
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = round(true_count / total_sample_count, 3)
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate(dataset):
    with tf.Graph().as_default() as g:
        # Get images and labels
        images, labels = reconobook_modelo.eval_inputs(dataset, FLAGS.batch_size, FLAGS.num_epochs)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = reconobook_modelo.inference(images)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(reconobook_modelo.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        summary_writer = tf.train.SummaryWriter(FLAGS.summary_dir, g)

        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):
    dataset = ReconoBookData(subset='validation')

    evaluate(dataset)


if __name__ == '__main__':
    tf.app.run()
