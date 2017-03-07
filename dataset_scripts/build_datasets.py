# coding=utf-8

# ==============================================================================
"""
------------------------------------------------------------------------------------------

DESCRIPCION:
    Convierte las imagenes en registros TF, donde cada registro tiene un objeto "example" serializado

ENTRADA:
    Las imagenes deben ser de formato JPG.
    Se espera que esten en el file system con la siguente estructura de carpetas:

      data_dir/label_0/image0.jpeg
      data_dir/label_0/image1.jpg
      ...
      data_dir/label_1/weird-image.jpeg
      data_dir/label_1/my-image.jpeg
      ...

    Donde el subdirectorio es la etiqueta unica asociada a cada imagen.

SALIDA:
    Este script convierte las imagenes en datasets particionados
    Cada dataset consiste de un conjunto de registros TF

        -------------------------
        train_directory/train-00000-of-01024
        train_directory/train-00001-of-01024
        ...
        train_directory/train-00127-of-01024
        -------------------------
        validation_directory/validation-00000-of-00128
        validation_directory/validation-00001-of-00128
        ...
        validation_directory/validation-00127-of-00128
        -------------------------

    En este caso se particiona el dataset de entrenamiento en 1024 partes
    y el dataset de validación en 128 partes

    La particion de data sets permite que varios hilos de ejecución puedan trabajar
    sobre el mismo conjunto de imagenes y distribuirse el trabajo para terminar
    más rapido la operación.

    Cada registro dentro del archivo tiene un objeto "example" serializado.
    El objeto example tiene los siguientes atributos:

        image/encoded: string containing JPEG encoded image in RGB colorspace
        image/height: integer, image height in pixels
        image/width: integer, image width in pixels
        image/colorspace: string, specifying the colorspace, always 'RGB'
        image/channels: integer, specifying the number of channels, always 3
        image/format: string, specifying the format, always'JPEG'
        image/filename: string containing the basename of the image file, e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
        image/class/label: integer specifying the index in a classification layer. The label ranges from [0, num_labels] where 0 is unused and left as the background class.
        image/class/text: string specifying the human-readable version of the label, e.g. 'dog'

    Observación:
    Un archivo TFRecords contiene una secuencia de caracteres con CRC hashes.
    Cada registro tiene el formato siguiente:

        - uint64 length
        - uint32 masked_crc32_of_length
        - byte   data[length]
        - uint32 masked_crc32_of_data

    y los registros se concatenan juntos para armar el archivo (dataset)


------------------------------------------------------------------------------------------
"""

# -----------------------------------------------------------------------------------------------

from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf

# -----------------------------------------------------------------------------------------------

tf.app.flags.DEFINE_string('train_directory', '../imagenes_jpg/train/', 'Directorio con las imagenes de entrenamiento')
tf.app.flags.DEFINE_string('validation_directory', '../imagenes_jpg/test/', 'Directorio con las imagenes de validación')
tf.app.flags.DEFINE_string('output_directory', '../datasets/', 'Directorio de salida')

tf.app.flags.DEFINE_integer('train_shards', 1, 'Numero de particiones del dataset de entrenamiento')
tf.app.flags.DEFINE_integer('validation_shards', 1, 'Numero de particiones del dataset de validación')

tf.app.flags.DEFINE_integer('num_threads', 1, 'Numero de hilos de ejecución')

# El archivo de labels (etiquetas) contiene una lista de las etiquetas validas
# El archivo contiene una lista de strings, donde cada linea corresponde a una etiqueta:
#   dog
#   cat
#   flower
# Se mapea la etiqueta al numero de linea donde está ubicado, comenzando desde el 0
tf.app.flags.DEFINE_string('labels_file', '../imagenes_jpg/labels.txt', 'Labels file')

FLAGS = tf.app.flags.FLAGS

# -----------------------------------------------------------------------------------------------


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg, feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})

        assert len(image.shape) == 3
        assert image.shape[2] == 3

        return image


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, text, height, width):
    """Build an Example proto for an example.
    Args:
        filename: string, path to an image file, e.g., '/path/to/example.JPG'
        image_buffer: string, JPEG encoding of RGB image
        label: integer, identifier for the ground truth for the network
        text: string, unique human-readable, e.g. 'dog'
        height: integer, image height in pixels
        width: integer, image width in pixels
    Returns:
        Example proto
    """

    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/text': _bytes_feature(text),
      'image/format': _bytes_feature(image_format),
      'image/filename': _bytes_feature(os.path.basename(filename)),
      'image/encoded': _bytes_feature(image_buffer)}))

    return example


def _is_png(filename):
    """Determine if a file contains a PNG format image.
    Args:
        filename: string, path of the image file.
    Returns:
        boolean indicating if the image is a PNG.
    """
    return '.png' in filename


def _process_image(filename, coder):
    """Process a single image file.
    Args:
        filename: string, path to an image file e.g., '/path/to/example.JPG'.
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
        image_buffer: string, JPEG encoding of RGB image.
        height: integer, image height in pixels.
        width: integer, image width in pixels.
    """
    # Leer el archivo de la imagen
    image_data = tf.gfile.FastGFile(filename, 'r').read()

    # Convertimos cualquier PNG a JPEG's para garantizar la consistencia
    if _is_png(filename):
        print('Convirtiendo PNG a JPEG para %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    # Decodificamos el RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Comprobamos que esté bien convertida a RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames, texts, labels, num_shards):
    """Procesa y guarga las imagenes como registros TF, en 1 hilo
    Args:
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
        thread_index: integer, unique batch to run index is within [0, len(ranges)).
        ranges: list of pairs of integers specifying ranges of each batches to analyze in parallel.
        name: string, unique identifier specifying the data set
        filenames: list of strings; each string is a path to an image file
        texts: list of strings; each string is human readable, e.g. 'dog'
        labels: list of integer; each integer identifies the ground truth
        num_shards: integer number of shards for this data set.
    """
    # Cada hilo produce N particiones, donde N = int(num_shards / num_threads).
    # Por ejemplo, si num_shards = 128, y el num_threads = 2, entonces cada hilo va a producir 64 particiones

    num_threads = len(ranges)
    # comprobamos que la cantidad de particiones total sea divisible por el numero de hilos total.
    assert not num_shards % num_threads
    # num_shards_per_batch: numero de particiones que genera cada hilo.
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                                ranges[thread_index][1],
                                num_shards_per_batch + 1).astype(int)

    # numero de archivos que va a procesar el hilo
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    # por cada particion a generar:
    for s in xrange(num_shards_per_batch):
        # Generamos el nombre del arhivo, que contenga la particion, ej: 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        # por cada archivo en la particion:
        for i in files_in_shard:
            # obtenemos los datos de la imagen
            filename = filenames[i]
            label = labels[i]
            text = texts[i]

            # procesamos la imagen
            image_buffer, height, width = _process_image(filename, coder)

            # generamos el objeto "example"
            example = _convert_to_example(filename, image_buffer, label, text, height, width)

            # lo serealizamos y lo escribimos en el archivo
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            # Cada 1000 imagenes imprimimos el progreso
            if not counter % 1000:
                print('%s [Hilo %d]: Procesadas %d de %d imagenes en el hilo batch' % (datetime.now(), thread_index, counter, num_files_in_thread))

            sys.stdout.flush()

        print('%s [Hilo %d]: Escribió %d imagenes en el archivo %s' % (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0

    print('%s [Hilo %d]: Escribió %d imagenes en %d particiones.' % (datetime.now(), thread_index, counter, num_files_in_thread))

    sys.stdout.flush()


def _process_image_files(name, filenames, texts, labels, num_shards):
    """Procesa y guarda el listado de imagenes como registros TF
    Args:
        name: string, identificador unico del dataset
        filenames: list of strings; cada string es la ruta a una imagen. (m*1) donde m es el numero de imagenes
        texts: list of strings; cada string es una etiqueta, por ejemplo 'dog' (m*1)
        labels: list of integer; cada entero identifica una etiqueta. (m*1)
        num_shards: integer, numero de particiones para este dataset
    """

    # comprobaciones
    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)

    # Separamos el listado de imagenes en varios rangos (varios batch), uno para cada hilo de ejecución
    spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = []

    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])

    # lanzamos un hilo de ejecución para cada batch
    print('Lanzando %d hilos para las siguientes separaciones del set de imagenes: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Creamos un macanismo para monitorear los hilos de ejecución
    coord = tf.train.Coordinator()

    # Instanciamos una utilidad generica de tensorFlow para codificar las imagenes
    coder = ImageCoder()

    # A hilo de ejecución le pasamos un batch
    threads = []
    for thread_index in xrange(len(ranges)):
        # definimos los parametros que le pasamos al metodo del hilo
        args = (coder, thread_index, ranges, name, filenames, texts, labels, num_shards)
        # creamos el hilo, va a ejecutar el método "_process_image_files_batch"
        t = threading.Thread(target=_process_image_files_batch, args=args)
        # iniciamos el hilo
        t.start()
        # guardamos el hilo en el listado de hilos
        threads.append(t)

    # Esperamos a que todos los hilos terminen:
    coord.join(threads)

    print('%s: Finalizada la escritura de %d imagenes en el data set.' % (datetime.now(), len(filenames)))

    sys.stdout.flush()


def _find_image_files(data_dir, labels_file):
    """ Genera una lista de todas las imagenes y etiquetas en el dataset
    Args:
        data_dir: string, ruta base a la carpeta donde se encuentran las imagenes
        labels_file: string, ruta al archivo de etiquetas.
    Returns:
        filenames: list of strings; cada string es la ruta a una imagen. (m*1) donde m es el numero de imagenes
        texts: list of strings; cada string es una etiqueta, por ejemplo 'dog' (m*1)
        labels: list of integer; cada entero identifica una etiqueta. (m*1)
    """

    print('Determinando la lista de imagenes y etiquetas desde: %s.' % data_dir)

    filenames = []
    texts = []
    labels = []

    # leemos el archivo de etiquetas
    unique_labels = [l.strip() for l in tf.gfile.FastGFile(labels_file, 'r').readlines()]

    # empezamos desde la etiqueta 0
    label_index = 0

    # construir la lista de imagenes y etiquetas.
    for text in unique_labels:
        # armamos la ruta donde estan las imagenes de esa etiqueta: data_dir/dog/*
        jpeg_file_path = '%s/%s/*' % (data_dir, text)
        # listamos los archivos de esa etiqueta
        matching_files = tf.gfile.Glob(jpeg_file_path)

        # agregamos los archivos con sus labels y text a los arrays
        labels.extend([label_index] * len(matching_files))
        texts.extend([text] * len(matching_files))
        filenames.extend(matching_files)

        print('Finalizado encontrar archivos de la clase %d, de un total de %d classes.' % (label_index, len(unique_labels)))

        label_index += 1

    # Mescla los registros de forma aleatoria, pero conservando la concordancia emtre ellos
    shuffled_index = range(len(filenames))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Encontrados %d imagenes JPEG, para %d etiquetas, en: %s.' % (len(filenames), len(unique_labels), data_dir))

    return filenames, texts, labels


def _process_dataset(name, directory, num_shards, labels_file):
    """Procesa un dataset completo y lo guarda.
    Args:
    name: string, identificador unico del dataset
    directory: string, directorio donde se encuentran las imagenes.
    num_shards: integer, Numero de particiones al dataset.
    labels_file: string, ruta del archivo de etiquetas.
    """

    # encontrar imagenes
    filenames, texts, labels = _find_image_files(directory, labels_file)

    # procesar imagenes
    _process_image_files(name, filenames, texts, labels, num_shards)


def main(unused_argv):
    assert not FLAGS.train_shards % FLAGS.num_threads, ('La cantidad de particiones (FLAGS.train_shards) debe ser divisible entre los hilos (FLAGS.num_threads)')

    assert not FLAGS.validation_shards % FLAGS.num_threads, ('La cantidad de particiones (FLAGS.validation_shards) debe ser divisible entre los hilos (FLAGS.num_threads)')

    print('Guardando dataset en: %s' % FLAGS.output_directory)

    # Generamos el dataset de validación
    _process_dataset('validation', FLAGS.validation_directory, FLAGS.validation_shards, FLAGS.labels_file)

    # Generamos el dataset de entrenamiento
    _process_dataset('train', FLAGS.train_directory, FLAGS.train_shards, FLAGS.labels_file)


# Punto de entrada del script
# tf.app.run() busca y ejecuta la función main del script
if __name__ == '__main__':
    tf.app.run()