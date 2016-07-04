# coding=utf-8

# ==============================================================================

"""
DESCRIPCION:
    Dada una carpeta con imagenes, divide la carpeta en 2 carpetas con imagenes, una para test y otra para train.

ENTRADA:
    La carpata debe tener la siguiente estructura:

          data_dir/label_0/image0.jpeg
          data_dir/label_0/image1.jpg
          ...
          data_dir/label_1/image0.jpeg
          data_dir/label_1/image1.jpeg

SALIDA:
    Al finalizar el proceso quedan las imagenes ordenadas de la siguiente manera:

          data_dir/train/label_0/image0.jpeg
          data_dir/train/label_0/image3.jpg
          ...
          data_dir/train/label_1/image1jpeg
          data_dir/train/label_1/image2.jpeg

          ...

          data_dir/test/label_0/image1.jpeg
          data_dir/test/label_0/image2.jpg
          ...
          data_dir/test/label_1/image0.jpeg
          data_dir/test/label_1/image3.jpg

    Ademas genera el archivo con los label en:

          data_dir/labels.txt

"""

# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import os.path
import shutil
from random import shuffle

# ==============================================================================

tf.app.flags.DEFINE_string('data_dir', '../imagenes_jpg/', 'Directorio con las imagenes')
tf.app.flags.DEFINE_string('train_folder', 'train', 'Directorio con las imagenes')
tf.app.flags.DEFINE_string('test_folder', 'test', 'Directorio con las imagenes')
tf.app.flags.DEFINE_string('labels_file_name', 'labels.txt', 'Labels file')
tf.app.flags.DEFINE_string('porcentaje_img_test', 40, 'Porcentaje de imagenes que van al set de test')

FLAGS = tf.app.flags.FLAGS

# ==============================================================================

def dividir_set(carpeta_raiz, subcarpeta, carpeta_train, carpata_test, porcentaje_img_test):
    # Listamos las imagenes
    imagenes = os.listdir(carpeta_raiz + subcarpeta)

    # Mesclamos aleatoriamente las imagenes
    shuffle(imagenes)

    # Calculamos la cantidad de imagenes de cada set
    cantidadTotalImagenes = len(imagenes)
    cantidadImagenesTest = int(cantidadTotalImagenes * porcentaje_img_test / 100)
    cantidadImagenesTrain = cantidadTotalImagenes - cantidadImagenesTest

    imagenesTrain = imagenes[0:cantidadImagenesTrain]
    imagenesTest = imagenes[cantidadImagenesTrain:cantidadTotalImagenes]

    for imagen in imagenesTrain:
        shutil.copy(carpeta_raiz + subcarpeta + "/" + imagen, carpeta_raiz + carpeta_train + "/" + subcarpeta + "/" + imagen)

    for imagen in imagenesTest:
        shutil.copy(carpeta_raiz + subcarpeta + "/" + imagen, carpeta_raiz + carpata_test + "/" + subcarpeta + "/" + imagen)


def obtenerSubCarpetas(carpeta):
    subcarpetas = [d for d in os.listdir(carpeta) if os.path.isdir(os.path.join(carpeta, d))]
    return subcarpetas


def main(unused_argv):
    print('Dividiendo set de imagenes de: %s' % FLAGS.data_dir)

    # Borramos las carpetas para el caso de que existan
    if os.path.exists(FLAGS.data_dir + FLAGS.train_folder):
        print('Borramos carpeta: %s' % FLAGS.data_dir + FLAGS.train_folder)
        shutil.rmtree(FLAGS.data_dir + FLAGS.train_folder)

    if os.path.exists(FLAGS.data_dir + FLAGS.test_folder):
        print('Borramos carpeta: %s' % FLAGS.data_dir + FLAGS.test_folder)
        shutil.rmtree(FLAGS.data_dir + FLAGS.test_folder)

    # Borramos el archivo de labels en caso de que exista:
    if os.path.isfile(FLAGS.data_dir + FLAGS.labels_file_name):
        os.remove(FLAGS.data_dir + FLAGS.labels_file_name)

    carpetas = obtenerSubCarpetas(FLAGS.data_dir)

    print('Encontradas las siguientes subcarpetas: %s' % carpetas)

    # creamos las carpetas
    os.mkdir(FLAGS.data_dir + FLAGS.train_folder)
    print('Creamos carpeta: %s' % FLAGS.data_dir + FLAGS.train_folder)
    os.mkdir(FLAGS.data_dir + FLAGS.test_folder)
    print('Creamos carpeta: %s' % FLAGS.data_dir + FLAGS.test_folder)

    # creamos el archivo de labels
    lablesFile = open(FLAGS.data_dir + FLAGS.labels_file_name, "a")

    esElPrimero = True
    for subcarpeta in carpetas:
        # creamos la subcarpeta
        os.mkdir(FLAGS.data_dir + FLAGS.train_folder + "/" + subcarpeta)
        os.mkdir(FLAGS.data_dir + FLAGS.test_folder + "/" + subcarpeta)

        print('Dividiendo imagenes de carpeta: %s' % subcarpeta)

        # copiar imagenes
        dividir_set(FLAGS.data_dir, subcarpeta, FLAGS.train_folder, FLAGS.test_folder, FLAGS.porcentaje_img_test)

        print('Divididas imagenes de carpeta %s' % subcarpeta)

        # Agregamos el label al archivo de labels
        if esElPrimero:
            lablesFile.write(subcarpeta)
            esElPrimero = False
        else:
            lablesFile.write("\n" + subcarpeta)

    # Cerramos el archivo
    lablesFile.close()

    print('Proceso finalizado')


# Punto de entrada del script
# tf.app.run() busca y ejecuta la función main del script
if __name__ == '__main__':
    tf.app.run()