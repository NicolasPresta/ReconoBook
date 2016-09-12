# coding=utf-8

# ==============================================================================

"""Read and preprocess image data.
 Image processing occurs on a single image at a time. Image are read and
 preprocessed in parallel across multiple threads. The resulting images
 are concatenated together to form a single batch for training or evaluation.
 -- Provide processed image data for a network:
 inputs: Construct batches of evaluation examples of images.
 distorted_inputs: Construct batches of training examples of images.
 batch_inputs: Construct batches of training or evaluation examples of images.
 -- Data processing:
 parse_example_proto: Parses an Example proto containing a training example
   of an image.
 -- Image decoding:
 decode_jpeg: Decode a JPEG encoded string into a 3-D float32 Tensor.
 -- Image preprocessing:
 image_preprocessing: Decode and preprocess one image for evaluation or training
 distort_image: Distort one image for training a network.
 eval_image: Prepare one image for evaluation.
 distort_color: Distort the color in one image for training.
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
                                      train=False,
                                      num_epochs=1)

    return images, labels


def eval_inputs(dataset, batch_size=None, num_epochs=1):
    """Generate batches of ImageNet images for evaluation.
    Use this function as the inputs for evaluating a network.
    Note that some (minimal) image preprocessing occurs during evaluation
    including central cropping and resizing of the image to fit the network.
    Args:
        dataset: instance of Dataset class specifying the dataset.
        batch_size: integer, number of examples in batch
        num_preprocess_threads: integer, total number of preprocessing threads but None defaults to FLAGS.num_preprocess_threads.
    Returns:
        images: Images. 4D tensor of size [batch_size, FLAGS.image_height, image_width, 3].
        labels: 1-D integer Tensor of [FLAGS.batch_size].
    """

    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.device('/cpu:0'):
        images, labels = batch_inputs(dataset,
                                      batch_size,
                                      train=False,
                                      num_epochs=num_epochs)

    return images, labels


def train_inputs(dataset, batch_size=None, num_epochs=1):
    """Generate batches of distorted versions of ImageNet images.
    Use this function as the inputs for training a network.
    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.
    Args:
    dataset: instance of Dataset class specifying the dataset.
    batch_size: integer, number of examples in batch
    num_preprocess_threads: integer, total number of preprocessing threads but
      None defaults to FLAGS.num_preprocess_threads.
    Returns:
    images: Images. 4D tensor of size [batch_size, FLAGS.image_height,
                                       FLAGS.image_width, 3].
    labels: 1-D integer Tensor of [batch_size].
    """

    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.device('/cpu:0'):
        images, labels = batch_inputs(dataset,
                                      batch_size,
                                      train=True,
                                      num_epochs=num_epochs)

    return images, labels


def decode_jpeg(image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.
    Args:
        image_buffer: scalar string Tensor.
        scope: Optional scope for op_scope.
    Returns:
        3-D float Tensor with values ranging from [0, 1).
    """
    with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
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


def distort_color(image, thread_id=0, scope=None):
    """Distort the color of the image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
        image: Tensor containing single image.
        thread_id: preprocessing thread ID.
        scope: Optional scope for op_scope.
    Returns:
        color-distorted image
    """

    with tf.op_scope([image], scope, 'distort_color'):
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


def distort_image(image, height, width, thread_id=0, scope=None):
    """Distort one image for training a network.
    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.
    Args:
        image: 3-D float Tensor of image
        height: integer
        width: integer
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged
            as [ymin, xmin, ymax, xmax].
        thread_id: integer indicating the preprocessing thread.
        scope: Optional scope for op_scope.
    Returns:
        3-D float Tensor of distorted image used for training.
    """

    with tf.op_scope([image, height, width], scope, 'distort_image'):

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

        # Randomly flip the image horizontally. (NMR No lo hacemos)
        # distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Randomly distort the colors.
        #distorted_image = distort_color(image, thread_id)

        # Subtract off the mean and divide by the variance of the pixels.
        #distorted_image = tf.image.per_image_whitening(distorted_image)

        return image


def eval_image(image, height, width, scope=None):
    """Prepare one image for evaluation.
    Args:
        image: 3-D float Tensor
        height: integer
        width: integer
        scope: Optional scope for op_scope.
    Returns:
        3-D float Tensor of prepared image.
    """
    with tf.op_scope([image, height, width], scope, 'eval_image'):
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
    """Decode and preprocess one image for evaluation or training.
    Args:
        image_buffer: JPEG encoded string Tensor
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged as
            [ymin, xmin, ymax, xmax].
        train: boolean
        thread_id: integer indicating preprocessing thread
    Returns:
        3-D float Tensor containing an appropriately scaled image
    Raises:
        ValueError: if user does not provide bounding box
    """

    image = decode_jpeg(image_buffer)

    height = FLAGS.image_height
    width = FLAGS.image_width

    # NMR: saco el preprocesado de la imagen con distorcion
    if train:
        image = distort_image(image, height, width, thread_id)
    else:
        image = eval_image(image, height, width)

    return image


def parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.
    The output of the build_datasets.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
        the following fields:
        image/height: 462
        image/width: 581
        image/colorspace: 'RGB'
        image/channels: 3
        image/class/label: 615
        image/class/text: 'knee pad'
        image/format: 'JPEG'
        image/filename: 'ILSVRC2012_val_00041207.JPEG'
        image/encoded: <JPEG encoded string>
    Args:
        example_serialized: scalar Tensor tf.string containing a serialized
                            Example protocol buffer.
    Returns:
        image_buffer: Tensor tf.string containing the contents of a JPEG file.
        label: Tensor tf.int32 containing the label.
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged as
            [ymin, xmin, ymax, xmax].
        text: Tensor tf.string containing the human-readable label.
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


def batch_inputs(dataset, batch_size, train, num_epochs=1):
    """Contruct batches of training or evaluation examples from the image dataset.
    Args:
        dataset: instance of Dataset class specifying the dataset. See base_dataset.py for details.
        batch_size: integer
        train: boolean
        num_preprocess_threads: integer, total number of preprocessing threads
        num_readers: integer, number of parallel readers
    Returns:
        images: 4-D float Tensor of a batch of images
        labels: 1-D integer Tensor of [batch_size].
    Raises:
        ValueError: if data is not found
    """
    with tf.name_scope('batch_processing'):

        # Obtenemos los nombres de los datasets que vamos a usar para leer los ejemplos
        data_files = dataset.data_files()
        if data_files is None:
            raise ValueError('No se especificó datasets')

        # Creamos el filename_queue
        if train:
            filename_queue = tf.train.string_input_producer(data_files, num_epochs=num_epochs, shuffle=True, capacity=8)
        else:
            filename_queue = tf.train.string_input_producer(data_files, shuffle=False, capacity=1)

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

        # Creamos el batch
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
