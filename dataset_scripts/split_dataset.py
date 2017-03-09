# coding=utf-8

# ==============================================================================

"""
DESCRIPCION:
    Dada una carpeta con imagenes, divide la carpeta en 2 carpetas con imagenes, una para validation y otra para train.

ENTRADA:
    La carpeta debe tener la siguiente estructura:

          img_dir/label_0/image0.jpeg
          img_dir/label_0/image1.jpg
          ...
          img_dir/label_1/image0.jpeg
          img_dir/label_1/image1.jpeg

SALIDA:
    Al finalizar el proceso quedan las imagenes ordenadas de la siguiente manera:

          img_dir/train/label_0/image0.jpeg
          img_dir/train/label_0/image3.jpg
          ...
          img_dir/train/label_1/image1jpeg
          img_dir/train/label_1/image2.jpeg

          ...

          img_dir/validation/label_0/image1.jpeg
          img_dir/validation/label_0/image2.jpg
          ...
          img_dir/validation/label_1/image0.jpeg
          img_dir/validation/label_1/image3.jpg

    Ademas genera el archivo con los label en:

          img_dir/labels.txt

"""

# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import glob
import shutil
import config
from random import shuffle

# ==============================================================================

FLAGS = tf.app.flags.FLAGS

capturasEntrenamiento = FLAGS.capturasEntrenamiento_id.split(",")
capturasTest = FLAGS.capturasTest_id.split(",")

# ==============================================================================


def dividir_set(carpeta_raiz, subcarpeta, carpeta_train, carpeta_validation, carpeta_test, porcentaje_img_validation):

    for captura in capturasEntrenamiento:
        # Listamos las imagenes
        imagenes = glob.glob(carpeta_raiz + "/" + subcarpeta + "/" + subcarpeta + captura + '*.jpg')

        # Mesclamos aleatoriamente las imagenes
        shuffle(imagenes)

        # Calculamos la cantidad de imagenes de cada set
        cantidadTotalImagenes = len(imagenes)
        cantidadImagenesValidation = int(cantidadTotalImagenes * porcentaje_img_validation / 100)
        cantidadImagenesTrain = cantidadTotalImagenes - cantidadImagenesValidation

        imagenesTrain = imagenes[0:cantidadImagenesTrain]
        imagenesValidation = imagenes[cantidadImagenesTrain:cantidadTotalImagenes]

        for imagen in imagenesTrain:
            name = os.path.basename(imagen)
            shutil.copy(carpeta_raiz + "/" + subcarpeta + "/" + name,
                        FLAGS.data_split_dir + "/" + carpeta_train + "/" + subcarpeta + "/" + name)

        for imagen in imagenesValidation:
            name = os.path.basename(imagen)
            shutil.copy(carpeta_raiz + "/" + subcarpeta + "/" + name,
                        FLAGS.data_split_dir + "/" + carpeta_validation + "/" + subcarpeta + "/" + name)

    for captura in capturasTest:
        # Listamos las imagenes
        imagenes = glob.glob(carpeta_raiz + "/" + subcarpeta + "/" + subcarpeta + captura + '*.jpg')
        for imagen in imagenes:
            name = os.path.basename(imagen)
            shutil.copy(carpeta_raiz + "/" + subcarpeta + "/" + name,
                        FLAGS.data_split_dir + "/" + carpeta_test + "/" + subcarpeta + "/" + name)

def obtenerSubCarpetas(carpeta):
    subcarpetas = [d for d in os.listdir(carpeta) if os.path.isdir(os.path.join(carpeta, d))]
    return subcarpetas


def main(unused_argv):
    print('Dividiendo set de imagenes de: %s' % FLAGS.img_dir)

    if os.path.exists(FLAGS.data_split_dir):
        shutil.rmtree(FLAGS.data_split_dir)
    if not os.path.exists(FLAGS.data_split_dir):
        os.mkdir(FLAGS.data_split_dir)

    # Borramos las carpetas para el caso de que existan
    if os.path.exists(FLAGS.data_split_dir + "/" + FLAGS.train_folder):
        print('Borramos carpeta: %s' % FLAGS.data_split_dir + "/" + FLAGS.train_folder)
        shutil.rmtree(FLAGS.data_split_dir + "/" + FLAGS.train_folder)

    if os.path.exists(FLAGS.data_split_dir + "/" + FLAGS.validation_folder):
        print('Borramos carpeta: %s' % FLAGS.data_split_dir + "/" + FLAGS.validation_folder)
        shutil.rmtree(FLAGS.data_split_dir + "/" + FLAGS.validation_folder)

    if os.path.exists(FLAGS.data_split_dir + "/" + FLAGS.test_folder):
        print('Borramos carpeta: %s' % FLAGS.data_split_dir + "/" + FLAGS.test_folder)
        shutil.rmtree(FLAGS.data_split_dir + "/" + FLAGS.test_folder)

    # Borramos el archivo de labels en caso de que exista:
    if os.path.isfile(FLAGS.img_dir + "/" + FLAGS.labels_file_name):
        os.remove(FLAGS.data_split_dir + "/" + FLAGS.labels_file_name)

    carpetas = obtenerSubCarpetas(FLAGS.img_dir)
    carpetas = map(int, carpetas)
    carpetas.sort()
    carpetas = map(str, carpetas)

    print('Encontradas las siguientes subcarpetas: %s' % carpetas)

    # creamos las carpetas
    os.mkdir(FLAGS.data_split_dir + "/" + FLAGS.train_folder)
    print('Creamos carpeta: %s' % FLAGS.data_split_dir + "/" + FLAGS.train_folder)
    os.mkdir(FLAGS.data_split_dir + "/" + FLAGS.validation_folder)
    print('Creamos carpeta: %s' % FLAGS.data_split_dir + "/" + FLAGS.validation_folder)
    os.mkdir(FLAGS.data_split_dir + "/" + FLAGS.test_folder)
    print('Creamos carpeta: %s' % FLAGS.data_split_dir + "/" + FLAGS.test_folder)

    # creamos el archivo de labels
    lablesFile = open(FLAGS.data_split_dir + "/" + FLAGS.labels_file_name, "a")

    esElPrimero = True
    for subcarpeta in carpetas:
        # creamos la subcarpeta
        os.mkdir(FLAGS.data_split_dir + "/" + FLAGS.train_folder + "/" + subcarpeta)
        os.mkdir(FLAGS.data_split_dir + "/" + FLAGS.validation_folder + "/" + subcarpeta)
        os.mkdir(FLAGS.data_split_dir + "/" + FLAGS.test_folder + "/" + subcarpeta)

        print('Dividiendo imagenes de carpeta: %s' % subcarpeta)

        # copiar imagenes
        dividir_set(FLAGS.img_dir, subcarpeta, FLAGS.train_folder, FLAGS.validation_folder,
                    FLAGS.test_folder, FLAGS.porcentaje_img_validation)

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
    os.chdir(os.getcwd() + "/..")
    tf.app.run()
