# coding=utf-8

# ==============================================================================

""" Evaluación del modelo ya entrenado """

# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import matplotlib.pyplot as plt
import tensorflow as tf
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


def load_image(filename):
    img = Image.open(filename)
    img.load()
    data = np.asarray(img, dtype="int32")
    if FLAGS.eval_distort:
        tf.set_random_seed(3251)
        data = tf.image.random_contrast(data, 0.5, 1)
        data = tf.image.random_hue(data, 0.15)
        data = tf.image.random_saturation(data, 0.5, 1)
    if FLAGS.eval_crop:
        data = tf.image.central_crop(data, (random.randint(7, 10)/10))
    data = tf.expand_dims(data, 0)
    data = tf.image.resize_bilinear(data, [FLAGS.image_height, FLAGS.image_width], align_corners=False)
    data = tf.image.convert_image_dtype(data, dtype=tf.float32)
    data = data.eval()
    # reescalamos la imagen
    data *= (1 / data.max())
    return data


def eval_once(saver, summary_writer, top_k_op, summary_op, datasetname, eval_num_examples, eval_batch_size):
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
            print('Modelo restaurado: global step = %s' % global_step)
        else:
            print('No checkpoint file found')
            return

        print("  ")
        print(" ------- ")
        print("--> Evaluando dataset: " + datasetname)
        print("--> Top K: " + str(FLAGS.top_k_prediction))
        print("--> eval_distort: " + str(FLAGS.eval_distort))
        print("--> eval_crop: " + str(FLAGS.eval_distort))
        print("--> eval_num_examples: " + str(eval_num_examples))
        print(" ------- ")
        print("  ")
                
        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = []
        try:
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_iter = int(math.ceil(eval_num_examples / eval_batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * eval_batch_size
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
            tf.summary.scalar('true_count', true_count)
            tf.summary.scalar('precision', precision)
            tf.summary.scalar('total_sample_count', total_sample_count)
            summary.value.add(tag=datasetname + 'Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate(datasetname, eval_num_examples, eval_batch_size):

    dataset = ReconoBookData(subset=FLAGS.eval_dataset)

    with tf.Graph().as_default() as g:
        # Obtenemos imagenes:
        images, labels = reconobook_modelo.eval_inputs(dataset, eval_batch_size)
        # image_shape = tf.reshape(images, [-1, FLAGS.image_height, FLAGS.image_width, 3])
        # tf.summary.image('input', image_shape, 3)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = reconobook_modelo.inference(images)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, FLAGS.top_k_prediction)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.summary_dir_eval, g)

        eval_once(saver, summary_writer, top_k_op, summary_op, datasetname, eval_num_examples, eval_batch_size)


def evaluate_unique(datasetname):
    dataset = ReconoBookData(subset=datasetname)

    with tf.Graph().as_default():

        # definimos placeholders
        _images = tf.placeholder(tf.float32, shape=[None, FLAGS.image_height, FLAGS.image_width, 3])

        # Obtenemos imagenes --(se cargan desde archivo, no desde dataset)
        if FLAGS.eval_unique_from_dataset:
            images, labels = reconobook_modelo.unique_input(dataset)
            # images, labels = reconobook_modelo.eval_inputs(dataset, 1, 1)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = reconobook_modelo.inference(_images)

        # Calculate predictions.
        maximaActivacion = tf.argmax(logits, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print('Modelo restaurado: global step = %s' % global_step)
            else:
                print('No checkpoint file found')
                return

            print("  ")
            print(" ------- ")
            print("--> eval_unique: " + str(FLAGS.eval_unique))
            print("--> eval_unique_from_dataset: " + str(FLAGS.eval_unique_from_dataset))
            print("--> Evaluando dataset: " + FLAGS.eval_dataset)
            print("--> eval_unique_cantidad_img: " + str(FLAGS.eval_unique_cantidad_img))
            print(" ------- ")
            print("  ")

            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = []
            try:
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

                for step in range(FLAGS.eval_unique_cantidad_img):

                    # cargamos imagen
                    if FLAGS.eval_unique_from_dataset:
                        imagenCargada = images.eval()
                    else:
                        imagenCargada = load_image(FLAGS.manual_test_folder + "%d.jpg" % (step + 1))

                    # La pasamos por el modelo de predicción
                    prediccion = sess.run([logits], feed_dict={_images: imagenCargada})

                    # Imprimios por consola
                    activaciones = prediccion[0][0]
                    activacionesDesc = np.sort(activaciones)[::-1]
                    top1Activacion = activacionesDesc[0]
                    top1Clase = np.where(activaciones == top1Activacion)[0]
                    top2Activacion = activacionesDesc[1]
                    top2Clase = np.where(activaciones == top2Activacion)[0]
                    top3Activacion = activacionesDesc[2]
                    top3Clase = np.where(activaciones == top3Activacion)[0]
                    print('-- EJEMPLO %d ----------------------------------------------------------------' % (step + 1))
                    print('Top 1 => Clase: %d, Activación: %s, Libro: %s' % (top1Clase[0],
                                                                             top1Activacion,
                                                                             titulos[top1Clase[0]]))
                    print('Top 2 => Clase: %d, Activación: %s, Libro: %s' % (top2Clase[0],
                                                                             top2Activacion,
                                                                             titulos[top2Clase[0]]))
                    print('Top 3 => Clase: %d, Activación: %s, Libro: %s' % (top3Clase[0],
                                                                             top3Activacion,
                                                                             titulos[top3Clase[0]]))
                    print('------------------------------------------------------------------------------')

                    for i in range(FLAGS.cantidad_clases):
                        print('Activación => Clase: %d, Activación: %s, Libro: %s' % (i, activaciones[i], titulos[i]))

                    print('------------------------------------------------------------------------------')
                    print('------------------------------------------------------------------------------')

                    # Mostramos la imagen
                    plt.imshow(imagenCargada[0, :, :, :], cmap='gray', interpolation='none')
                    plt.show()

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


def main(argv=None):
    
    # creamos el directorio de checkpoint_dir si no existe, y si existe lo borramos y creamos de nuevo
    if not os.path.exists(FLAGS.summary_dir_eval):
        os.mkdir(FLAGS.summary_dir_eval)
    # else:
    #    shutil.rmtree(FLAGS.summary_dir_eval)
    #    os.mkdir(FLAGS.summary_dir_eval)

    if FLAGS.eval_unique:
        evaluate_unique(FLAGS.eval_dataset)
    else:
        evaluate(FLAGS.eval_dataset, FLAGS.eval_num_examples, FLAGS.eval_batch_size)

    input = sys.stdin.readline()

if __name__ == '__main__':
    tf.app.run()
