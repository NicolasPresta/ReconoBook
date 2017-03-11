# coding=utf-8

# ==============================================================================

"""Definición del modelo"""

# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import reconobook_input
import re
import config

# ==============================================================================

FLAGS = tf.app.flags.FLAGS

# ==============================================================================


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


# Funciones utiles para inicializar parametros
def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """
    var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')

    tf.add_to_collection('losses', weight_decay)

    return var


# Funciones utiles para realizar operaciones
def _conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def _max_pool_2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


# Funciones para obtener datos que alimenten el modelo
def train_inputs(dataset, batchSize):
    return reconobook_input.train_inputs(dataset, batch_size=batchSize)


def eval_inputs(dataset, batchSize):
    return reconobook_input.eval_inputs(dataset, batch_size=batchSize)


def unique_input(dataset):
    return reconobook_input.unique_input(dataset)


# Armado del modelo:
def inference(images):
    """ Armamos el modelo.
    Contrucción de la red neuronal profunda
    Args:
        images: Images returned from distorted_inputs() or inputs().
    Returns:
        Logits.
    """

    # Primer capa convolucional
    with tf.name_scope("CONV-1"):
        kernels_conv1 = _variable_with_weight_decay("kernels_conv1", [5, 5, 3, FLAGS.model_cant_kernels1], stddev=1e-4, wd=0.004)
        bias_conv1 = tf.get_variable("bias_conv1", [FLAGS.model_cant_kernels1], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.relu(_conv2d(images, kernels_conv1) + bias_conv1, name="conv1")

    # max pool 1
    with tf.name_scope("MAXPOOL-1"):
        pool1 = _max_pool_2x2(conv1, "pool1")

    # normalización 1
    with tf.name_scope("NORM-1"):
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # Segunda capa convolucional
    with tf.name_scope("CONV-2"):
        kernels_conv2 = _variable_with_weight_decay("kernels_conv2", shape=[3, 3, FLAGS.model_cant_kernels1, FLAGS.model_cant_kernels2], stddev=1e-4, wd=0.004)
        bias_conv2 = tf.get_variable("bias_conv2", [FLAGS.model_cant_kernels2], initializer=tf.constant_initializer(0.1))
        conv2 = tf.nn.relu(_conv2d(norm1, kernels_conv2) + bias_conv2, name="conv2")

    # normalización 2
    with tf.name_scope("NORM-2"):
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    # max pool 1
    with tf.name_scope("MAXPOOL-2"):
        pool2 = _max_pool_2x2(norm2, "pool2")

    # primer capa full conected
    with tf.name_scope("FC-1"):
        pool2_flat = tf.reshape(pool2, [-1, 10 * 10 * FLAGS.model_cant_kernels2])
        W_fc1 = _variable_with_weight_decay("W_fc1", shape=[10 * 10 * FLAGS.model_cant_kernels2, FLAGS.model_cant_fc1], stddev=0.04, wd=0.004)
        b_fc1 = tf.get_variable("b_fc1", [FLAGS.model_cant_fc1], initializer=tf.constant_initializer(0.1))
        local1 = tf.nn.relu(tf.matmul(pool2_flat, W_fc1) + b_fc1, name="local1")

    # segunda capa full conected
    with tf.name_scope("FC-2"):
        W_fc2 = _variable_with_weight_decay("W_fc2", shape=[FLAGS.model_cant_fc1, FLAGS.cantidad_clases], stddev=0.04, wd=0.004)
        b_fc2 = tf.get_variable("b_fc2", [FLAGS.cantidad_clases], initializer=tf.constant_initializer(0.1))
        logits = tf.matmul(local1, W_fc2) + b_fc2

    return logits


def loss(logits, labels):
    """Cada un conjunto de predicciones y de etiquetas, retorna el costo de la predicción.
    Args:
        logits: Logits retornados por inference().
        labels: Labels reales de las imagenes
    Returns:
        Loss tensor del tipo float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(total_loss, global_step):
    """Entrenamiento del modelo
    Crea un optimizador y lo aplica a todas las variables
    Args:
        total_loss: costo total desde loss()
        global_step: variable que cuenta la cantidad de pasos de entrenamiento
    Returns:
        train_op: operación para entrenar.
    """

    # Reducimos el learning rate exponencialmente dependiendo el número de pasos de entrenamiento
    lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step,
                                    FLAGS.decay_steps,
                                    FLAGS.decay_rate,
                                    staircase=True)

    # Definimos el optimizador a utilizar
    opt = tf.train.GradientDescentOptimizer(lr)

    # Agrega summaries
    tf.summary.scalar('learning_rate', lr)
    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        # Computa los gradientes
        grads = opt.compute_gradients(total_loss)
        # aplica los gradientes.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Agrega el histograma para las variables de entrenamiento
    #for var in tf.trainable_variables():
    #    tf.summary.histogram(var.op.name, var)

    # Agrega el histograma para los gradientes
    #for grad, var in grads:
    #    if grad is not None:
    #        tf.summary.histogram(var.op.name + '/gradients', grad)

    # guardamos el promedio movil de las variables, es util para mejorar la eficiencia del optimizador.
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op



