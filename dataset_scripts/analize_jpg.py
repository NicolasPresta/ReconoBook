# coding=utf-8

# ==============================================================================

"""
DESCRIPCION:
    Analiza la composición de imagenes de cada clase
    Empareja las observaciones por clase para que todas las clases tengan
    la misma cantidad de imagenes por cada captura.

ENTRADA:
    La carpata debe tener la siguiente estructura:

          data_dir/label_0/A image0.jpeg
          data_dir/label_0/B image1.jpg
          ...
          data_dir/label_1/A image0.jpeg
          data_dir/label_1/B image1.jpeg

SALIDA:
    Una tabla sumarizando la cantidad de imagen por cada clase, por cada captura.

"""
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import os.path
import glob
from random import shuffle

# ==============================================================================

tf.app.flags.DEFINE_string('data_dir', '../imagenes_jpg/', 'Directorio con las imagenes')
tf.app.flags.DEFINE_integer('img_por_captura', 110, 'Cantidad de imagenes a concervar por captura')
tf.app.flags.DEFINE_boolean('modo_procesar', 0, 'Indica si en lugar de hacer el analisis se realiza la limpieza')

capturas = ["A", "B", "C"]

FLAGS = tf.app.flags.FLAGS

# ==============================================================================


def obtenerSubCarpetas(carpeta):
    subcarpetas = [d for d in os.listdir(carpeta) if os.path.isdir(os.path.join(carpeta, d))]
    return subcarpetas


def main(unused_argv):
    if FLAGS.modo_procesar:
        procesar()
    else:
        analisar()


def analisar():
    print('Analisando imagenes de: %s' % FLAGS.data_dir)

    carpetas = obtenerSubCarpetas(FLAGS.data_dir)

    encabezado = "clase\t"
    for captura in capturas:
        encabezado = encabezado + captura + "\t\t"
    encabezado = encabezado + "Total"

    print(encabezado)

    for subcarpeta in carpetas:
        renglon = subcarpeta + "\t\t"
        carpeta = FLAGS.data_dir + subcarpeta + "/"

        total = 0
        for captura in capturas:
            capturaX = glob.glob(carpeta + subcarpeta + captura + "*.jpg")
            cantCapX = len(capturaX)
            total = total + cantCapX
            renglon = renglon + str(cantCapX) + "\t\t"

        renglon = renglon + str(total)

        print(renglon)

    print('Proceso finalizado')


def procesar():
    print('Procesando imagenes de: %s' % FLAGS.data_dir)

    carpetas = obtenerSubCarpetas(FLAGS.data_dir)

    for subcarpeta in carpetas:

        carpeta = FLAGS.data_dir + subcarpeta + "/"

        for captura in capturas:
            capturaX = glob.glob(carpeta + subcarpeta + captura + "*.jpg")

            # Mesclamos los archivos aleatoriamente
            shuffle(capturaX)

            # Nos quedamos con los primeros img_por_captura elementos
            capturaX = capturaX[0:FLAGS.img_por_captura]

            # Borramos aquellas imagenes que no estén dentro de captura
            for archivo in glob.glob(carpeta + subcarpeta + captura + "*.jpg"):
                if not (archivo in capturaX):
                    os.remove(archivo)

    print('Proceso finalizado')
    analisar()


# Punto de entrada del script
# tf.app.run() busca y ejecuta la función main del script
if __name__ == '__main__':
    tf.app.run()