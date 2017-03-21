# coding=utf-8

# ==============================================================================

""" Evaluación del modelo ya entrenado """

# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from itertools import groupby
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
        data = tf.image.central_crop(data, (random.randint(7, 10) / 10))
    data = tf.expand_dims(data, 0)
    data = tf.image.resize_bilinear(data, [FLAGS.image_height, FLAGS.image_width], align_corners=False)
    data = tf.image.convert_image_dtype(data, dtype=tf.float32)
    data = data.eval()
    # reescalamos la imagen
    data *= (1 / data.max())
    return data


def RestaurarModelo(sess):
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        return global_step
    else:
        return None


def ObtenerImagenes(datasetname, eval_num_examples):
    with tf.Graph().as_default() as g:
        dataset = ReconoBookData(subset=datasetname)
        images, labels = reconobook_modelo.eval_inputs(dataset, eval_num_examples)
        with tf.Session() as sess:
            # Restore del modelo guardado (checkpoint)
            #global_step = RestaurarModelo(sess)
            coord = tf.train.Coordinator()
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            # Obtenemos imagenes:
            x_images, x_labels = sess.run([images, labels])

            # Cerramos los hilos de lectura
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

            return x_images, x_labels


def EvalularImagenes(images, labels, eval_num_examples):
    with tf.Graph().as_default() as g:
        # Definimos los placeholder para el input
        _images = tf.placeholder(tf.float32, shape=[eval_num_examples, FLAGS.image_height, FLAGS.image_width, 3])
        _labels = tf.placeholder(tf.int32, shape=[eval_num_examples])

        # Le pasamos las imagenes al modelo para que nos de su predicción
        logits = reconobook_modelo.inference(_images)

        # Calculamos si están en el top k
        top_k_op = tf.nn.in_top_k(logits, _labels, FLAGS.top_k_prediction)

        # Build the summary operation based on the TF collection of Summaries.
        run_options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
        run_metadata = tf.RunMetadata()
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.summary_dir_eval, g)

        with tf.Session() as sess:
            # Restore del modelo guardado (checkpoint)
            global_step = RestaurarModelo(sess)
            if global_step:
                print('Modelo restaurado: global step = %s' % global_step)
            else:
                print('No se puede restaurar modelo')

            # Evaluamos las imagenes
            predictions, activaciones = sess.run([top_k_op, logits],
                                                feed_dict={_images: images, _labels: labels},
                                                run_metadata=run_metadata,
                                                options=run_options)

            # Guardamos el summary
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op, feed_dict={_images: images, _labels: labels}))
            summary_writer.add_summary(summary, global_step)
            summary_writer.add_run_metadata(run_metadata, 'step' + global_step)

            # Summary de imagenes erroneas

            # clases = tf.argmax(logits, axis=1)
            # for i in range(FLAGS.eval_num_examples):
            #    tf.summary.image(str(clases[i].eval())+'_clase', tf.expand_dims(images[i, :, :, :], 0))

            # to tf.image_summary format [batch_size, height, width, channels]
            # summaryImg = tf.Summary.Image()
            # for i in range(eval_num_examples):
            #    if predictions[0][i]:
            #        summary = tf.Summary()
            #        tf.summary.image('img_ok_' + str(i), images[i, :, :, :], 1)
            #    else:
            #        tf.summary.image('img_error_' + str(i), images[i, :, :, :], 1)


            # summary_writer.add_summary(summary, global_step)
            # summary_writer.add_summary(summaryImg, global_step)

            return global_step, predictions, activaciones


def ImprimirEncabezado(datasetname, eval_num_examples):
    # Imprimimos resumen de la configuración de evaluación
    print("  ")
    print(" ------- ")
    print("--> Evaluando dataset: " + datasetname)
    print("--> Top K: " + str(FLAGS.top_k_prediction))
    print("--> eval_distort: " + str(FLAGS.eval_distort))
    print("--> eval_crop: " + str(FLAGS.eval_distort))
    print("--> eval_num_examples: " + str(eval_num_examples))
    print(" ------- ")
    print("  ")


def ImprimirResumen(predictions, labels, datasetname, global_step):

    with tf.Graph().as_default() as g:
        summary = tf.Summary()
        summary_writer = tf.summary.FileWriter(FLAGS.summary_dir_eval, g)

        predictions = predictions[0]

        # Total Count
        total_count = len(predictions)

        # Totalizamos el númer de predicciones correctas
        true_count = np.sum(predictions)

        # Calculamos la precisión
        precision = round(true_count / total_count, 3)

        summary.value.add(tag='precision_' + datasetname, simple_value=precision)

        # Imprimimos resultados por pantalla
        print(" ------- ")
        print('true_count = %d' % true_count)
        print('total_sample_count = %d' % total_count)
        print('precision = %.3f' % precision)
        print(" ------- ")

        true_labels = labels[predictions]
        false_labels = labels[[not i for i in predictions]]
        print(" ------- ")
        print("clase\ttotal\ttrue\tfalse\tpresicion\tlibro")
        clase = 0
        for x in range(FLAGS.cantidad_clases):
            true_count = (true_labels == clase).sum()
            false_count = (false_labels == clase).sum()
            total_count = true_count + false_count
            if total_count == 0:
                total_count = 1
            pres = round(true_count / total_count, 2)
            print("%d\t\t%d\t\t%d\t\t%d\t\t%.2f\t\t%s" % (clase+1, total_count, true_count, false_count, pres, titulos[clase]))
            summary.value.add(tag=str(clase+1) + '_precision_' + datasetname, simple_value=pres)
            clase = clase + 1

        print(" ------- ")

        # Guardamos en el summary el resultado
        summary_writer.add_summary(summary, global_step)


def evaluate(datasetname, eval_num_examples):
    # Mostramos por consola la configuración de evaluación
    ImprimirEncabezado(datasetname, eval_num_examples)

    # Input de imagenes
    images, labels = ObtenerImagenes(datasetname, eval_num_examples)

    # Calcular si la predicción fue correcta o no
    global_step, predictions, _ = EvalularImagenes(images, labels, eval_num_examples)

    # Imprime por consola y en el summary el resultado
    ImprimirResumen(predictions, labels, datasetname, global_step)


def evaluate_unique(datasetname):
    # Input de imagenes
    images, labels = ObtenerImagenes(datasetname, FLAGS.eval_unique_cantidad_img)
    # Calcular si la predicción fue correcta o no
    global_step, _, activ = EvalularImagenes(images, labels, FLAGS.eval_unique_cantidad_img)

    for step in range(FLAGS.eval_unique_cantidad_img):
        activaciones = activ[step, :]
        # Imprimios por consola
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
        plt.imshow(images[step, :, :, :], cmap='gray', interpolation='none')
        plt.show()

    print('---- Fin de la evaluación')


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
        evaluate(FLAGS.eval_dataset, FLAGS.eval_num_examples)

    input = sys.stdin.readline()


if __name__ == '__main__':
    tf.app.run()
