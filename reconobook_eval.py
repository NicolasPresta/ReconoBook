# coding=utf-8

# ==============================================================================

""" Evaluación del modelo ya entrenado """

# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from reconobook_dataset import ReconoBookData
import reconobook_modelo
import PIL
from PIL import Image

# ==============================================================================

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summary_dir', './summary_eval', "Directory where to write event logs.")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5, "How often to run the eval.")
tf.app.flags.DEFINE_integer('num_examples', 2000, "Number of examples to run.")
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints', "Directory where to write checkpoint.")
tf.app.flags.DEFINE_boolean('run_once', True, "Whether to run eval only once.")
tf.app.flags.DEFINE_boolean('unique', True, "Ejecutar revisión imagen por imagen")
tf.app.flags.DEFINE_integer("batch_size", 100, "Cantidad de imagenes que se procesan en un batch")
tf.app.flags.DEFINE_integer('num_epochs', 1, 'Cantidad de epocas')

# ==============================================================================


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


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
            print('true_count = %d' % true_count)
            print('total_sample_count = %d' % total_sample_count)
            print('precision = %.3f' % precision)

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
        # Obtenemos imagenes:
        images, labels = reconobook_modelo.eval_inputs(dataset, FLAGS.batch_size, FLAGS.num_epochs)
        image_shape = tf.reshape(images, [-1, 28, 28, 3])
        tf.image_summary('input', image_shape, 3)

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


def evaluate_unique(dataset):
    with tf.Graph().as_default():
        # definimos placeholders
        _images = tf.placeholder(tf.float32, shape=[None, 28, 28, 3])
        #_labels = tf.placeholder(tf.int32, shape=[None])

        # Obtenemos imagenes:
        images, labels = reconobook_modelo.unique_input(dataset)
        # images, labels = reconobook_modelo.eval_inputs(dataset, 1, 1)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = reconobook_modelo.inference(_images)

        # Calculate predictions.
        # top_k_op = tf.nn.in_top_k(logits, _labels, 1)
        maximaActivacion = tf.argmax(logits, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(reconobook_modelo.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print('global step = %s' % global_step)
            else:
                print('No checkpoint file found')
                return


            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = []
            try:
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

                for step in xrange(50):
                    var1 = images.eval()
                    #var1 = load_image("./imagenes_jpg/1/1 001.jpg")
                    #var1 = np.reshape(var1, (28, 28, 3))
                    #var1 = var1.resize((28, 28), Image.ANTIALIAS)
                    prediccion = sess.run([maximaActivacion], feed_dict={_images: var1})
                    titulo = ""

                    if prediccion[0] == 0:
                        titulo = "Einstein"
                    if prediccion[0] == 1:
                        titulo = "Analisis matematico (Volumen 2)"
                    if prediccion[0] == 2:
                        titulo = "Sistemas inteligentes"
                    if prediccion[0] == 3:
                        titulo = "Mineria de datos a traves de ejemplos"
                    if prediccion[0] == 4:
                        titulo = "Analisis matematico (Volumen 3)"
                    if prediccion[0] == 5:
                        titulo = "Introducción a la Mineria de datos"
                    if prediccion[0] == 6:
                        titulo = "Big data"
                    if prediccion[0] == 7:
                        titulo = "Patrones de diseño"
                    if prediccion[0] == 8:
                        titulo = "Fisica universitaria"
                    if prediccion[0] == 9:
                        titulo = "Sistemas expertos"

                    print('Predicción = %s, Titulo: %s' % (prediccion, titulo))
                    plt.imshow(var1[0, :, :, :], cmap='gray', interpolation='none')
                    plt.show()

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


def main(argv=None):
    dataset = ReconoBookData(subset='validation')

    if FLAGS.unique:
        evaluate_unique(dataset)
    else:
        evaluate(dataset)

if __name__ == '__main__':
    tf.app.run()
