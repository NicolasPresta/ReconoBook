# coding=utf-8

# ==============================================================================

"""Lectura y preprocesado de una imagen.
    El preprocesado se realiza en varios hilos, cada hilo preprocesa una unica imagen
    por vez.
    Las imagenes se levantan desde el DataSet, (no se leen directamente los JPG)

    -- Input de datos para la red neuronal:
        unique_input: retorna una única imagen del dataset. Es útil para evaluar imagenes de a una a la vez.
        eval_inputs: Lotes de imagenes de evaluación
        train_inputs: Lotes de imagenes de entrenamiento.
        batch_inputs: Construye los lotes de imagenes para los demás metodos.

"""

# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import random
import config

# ==============================================================================

FLAGS = tf.app.flags.FLAGS

# ==============================================================================


def unique_input(dataset):
    with tf.device('/cpu:0'):
        images, labels = batch_inputs(dataset,
                                      batch_size=1,
                                      train=False)

    return images, labels


def eval_inputs(dataset, batch_size=None):
    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.device('/cpu:0'):
        images, labels = batch_inputs(dataset,
                                      batch_size,
                                      train=False)

    return images, labels


def train_inputs(dataset, batch_size=None):
    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.device('/cpu:0'):
        images, labels = batch_inputs(dataset,
                                      batch_size,
                                      train=True)

    return images, labels


def batch_inputs(dataset, batch_size, train):
    """Contruye lotes de imagenes
    Args:
        dataset: Un objeto de la clase ReconoBookData
        batch_size: Cantidad de imagenes que tiene el lote
        train: boolean, indice si las imagenes son para entrenamiento o para evaluación
    Returns:
        images: 4-D float Tensor
        labels: 1-D integer Tensor
    Raises:
        ValueError: Si no hay dataset.
    """

    with tf.name_scope('batch_processing'):

        # Obtenemos la ruta del dataset que vamos a usar para leer los ejemplos
        data_files = dataset.data_files()
        if data_files is None:
            raise ValueError('No se especificó la ruta del dataset')

        # Creamos el filename_queue,
        if train:
            filename_queue = tf.train.string_input_producer(data_files, shuffle=True, capacity=8)
        else:
            filename_queue = tf.train.string_input_producer(data_files, shuffle=False, capacity=8)

        # Creamos el reader
        reader = dataset.reader()
        _, example_serialized = reader.read(filename_queue)

        # Preprocesado
        images_and_labels = []
        for thread_id in range(FLAGS.input_num_preprocess_threads):
            # Deserealizamos el string con el objeto example
            image_buffer, label_index, _ = parse_example_proto(example_serialized)
            # Preprocesamos la imagen
            image = image_preprocessing(image_buffer, train, thread_id)
            # La agregamos al listado
            images_and_labels.append([image, label_index])

        # Creamos el batch, donde confluyen los distintos hilos de preprocesado
        images, label_index_batch = tf.train.batch_join(images_and_labels,
                                                        batch_size=batch_size,
                                                        capacity=2 * FLAGS.input_num_preprocess_threads * batch_size)

        # Redimensionamos a las dimensiones establecidas
        height = FLAGS.image_height
        width = FLAGS.image_width
        depth = 3

        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[batch_size, height, width, depth])

        return images, tf.reshape(label_index_batch, [batch_size])


def decode_jpeg(image_buffer):
    """Decodifica una imagen JPG en string en un 3-D float image Tensor.
    Args:
        image_buffer: scalar string Tensor.
    Returns:
        3-D float Tensor con valores entre [0, 1).
    """
    with tf.name_scope(values=[image_buffer], name='decode_jpeg'):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_jpeg(image_buffer, channels=3)

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).  The various
        # adjust_* ops all require this range for dtype float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        return image


def distort_color(image, thread_id=0):
    """Distorsiona el color de una imagen
    Cada distorción de color no es conmutativa, por lo que el orden importa.
    Idealmente el orden de aplicación de las distorciones de color debería ser aleatoria.
    Pero es suficientemente buena solución si alteramos el orden de aplicación de las
    distorciones en función del hilo que ejecuta este preprocesado.
    Args:
        image: Tensor conteniendo una sola imagen.
        thread_id: preprocessing thread ID.
    Returns:
        imagen distorcionada
    """

    with tf.name_scope(values=[image], name='distort_color'):
        color_ordering = thread_id % 2

        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)

        return image


def distort_image(image, height, width):
    """Distorciona la imagen para el entrenamiento de la red
    El distorsionado de imagen es una forma efectiva de generar más imagenes distintas
    y evitar el sobre ajuste de nuestra red neuronal
    Args:
        image: 3-D float Tensor
        height: integer
        width: integer
    Returns:
        3-D float Tensor
    """

    with tf.name_scope(values=[image, height, width], name='distort_image'):

        # distort the image
        if FLAGS.train_distort:
            image = tf.image.random_contrast(image, 0.5, 1)
            image = tf.image.random_hue(image, 0.15)
            image = tf.image.random_saturation(image, 0.5, 1)

        if FLAGS.train_crop:
            image = tf.image.central_crop(image, (random.randint(7, 10) / 10))

        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
        image = tf.squeeze(image, [0])

        # Las siguientes distorciones no se realizan por el momento.
        # Randomly flip the image horizontally.
        # distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Randomly distort the colors.
        #distorted_image = distort_color(image, thread_id)

        # Subtract off the mean and divide by the variance of the pixels.
        #distorted_image = tf.image.per_image_whitening(distorted_image)

        return image


def eval_image(image, height, width):
    """Prepara una imagen para la evaluación.
    Dependiendo de lo configurado puede distorsionarse o no.
    Args:
        image: 3-D float Tensor
        height: integer
        width: integer
    Returns:
        3-D float Tensor
    """
    with tf.name_scope(values=[image, height, width], name='eval_image'):
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        # image = tf.image.central_crop(image, central_fraction=0.875)

        if FLAGS.eval_distort:
            image = tf.image.random_contrast(image, 0.5, 1)
            image = tf.image.random_hue(image, 0.15)
            image = tf.image.random_saturation(image, 0.5, 1)

        if FLAGS.eval_crop:
            image = tf.image.central_crop(image, (random.randint(7, 10) / 10))

        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
        image = tf.squeeze(image, [0])

        return image


def image_preprocessing(image_buffer, train, thread_id=0):
    """Defodifica y preprocesa una imagen para evaluación o entrenamiento
    Args:
        image_buffer: JPEG encoded string Tensor
        train: boolean
        thread_id: el número de hilo de preprocesado
    Returns:
        3-D float Tensor
    """

    image = decode_jpeg(image_buffer)

    height = FLAGS.image_height
    width = FLAGS.image_width

    if train:
        image = distort_image(image, height, width)
    else:
        image = eval_image(image, height, width)

    return image


def parse_example_proto(example_serialized):
    """Parsea una linea del dataset que contiene una solo imagen
    La salida del script build_datasets.py es un conjunto de objetos proto.
    Args:
        example_serialized: string leido del dataset.
    Returns:
        image_buffer: Tensor tf.string
        label: Tensor tf.int32
        text: Tensor tf.string coneniendo el label leible por humanos
    """

    # Especificamos que atributos queremos deserealizar
    feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
    }

    # las deserealizamos
    features = tf.parse_single_example(example_serialized, feature_map)

    # Obtenemos el label como Int32
    label = tf.cast(features['image/class/label'], dtype=tf.int32)
    labelText = features['image/class/text']
    imagen = features['image/encoded']

    return imagen, label, labelText


