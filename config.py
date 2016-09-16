# coding=utf-8

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Directorios
tf.app.flags.DEFINE_string('summary_dir_eval', './summary_eval', "Logs de proceso de evaluación")
tf.app.flags.DEFINE_string('summary_dir_train', './summary_train', "Logs de proceso de entrenamiento")
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints', "Resguardo del modelo a utilizar")
tf.app.flags.DEFINE_string('data_dir', './datasets', 'Directorio donde estan los datasets a utilizar')

# Dataset
tf.app.flags.DEFINE_integer('cantidad_imagenes_train', 4631, 'Cantidad de imagenes distintas para entrenamiento')
tf.app.flags.DEFINE_integer('cantidad_imagenes_eval', 3073, 'Cantidad de imagenes distintas para evaluar modelo')
tf.app.flags.DEFINE_integer('cantidad_clases', 20, 'Cantidad de clases a reconocer')

# Imagen
tf.app.flags.DEFINE_integer('image_height', 40, 'Alto imagen')
tf.app.flags.DEFINE_integer('image_width', 40, 'Ancho imagen')

# Evaluación
tf.app.flags.DEFINE_boolean('eval_unique', True, "Ejecutar revisión imagen por imagen")
tf.app.flags.DEFINE_boolean('eval_unique_from_dataset', False, "Evaluar imagen por imagen desde dataset")
tf.app.flags.DEFINE_integer('eval_unique_cantidad_img', 30, "Cantidad de imagenes a evaluar si eval_unique = true")
tf.app.flags.DEFINE_boolean('eval_distort', True, "Distorcionar imagenes al evaluar")
tf.app.flags.DEFINE_boolean('eval_crop', True, "Distorcionar imagenes al evaluar")
tf.app.flags.DEFINE_integer('eval_num_examples', 1000, "Número de imagenes a evaluar")
tf.app.flags.DEFINE_integer("eval_batch_size", 100, "Cantidad de imagenes que se evaluan por batch")

# Entrenamiento
tf.app.flags.DEFINE_integer("moving_average_decay", 0.9999, "The decay to use for the moving average.")
tf.app.flags.DEFINE_integer("initial_learning_rate", 0.1, "Initial learning rate.")
tf.app.flags.DEFINE_integer("decay_steps", 500, "Epochs after which learning rate decays.")
tf.app.flags.DEFINE_integer("decay_rate", 0.9, "Learning rate decay factor.")
tf.app.flags.DEFINE_boolean('log_device_placement', False, "Whether to log device placement.")
tf.app.flags.DEFINE_boolean('train_distort', True, "Distorcionar imagenes al evaluar")
tf.app.flags.DEFINE_boolean('train_crop', True, "Distorcionar imagenes al evaluar")
tf.app.flags.DEFINE_integer('train_max_steps', 20000, "Number of batches to run.")
tf.app.flags.DEFINE_integer("train_batch_size", 100, "Cantidad de imagenes que se procesan por batch")
tf.app.flags.DEFINE_integer('train_num_epochs', 500, 'Cantidad de epocas')
# OJO: (cantidad_imagenes_train) * (num_epochs) > (batch_size) * (max_steps)

# Input
tf.app.flags.DEFINE_integer('input_num_preprocess_threads', 4, 'Numero de hilos que hacen el preprocesado')
tf.app.flags.DEFINE_integer('input_num_readers', 4, 'Numero de readers')

# Modelo
tf.app.flags.DEFINE_integer('model_cant_kernels1', 30, 'Cantidad de kernels de convolución en la capa 1')
tf.app.flags.DEFINE_integer('model_cant_kernels2', 60, 'Cantidad de kernels de convolución en la capa 2')
tf.app.flags.DEFINE_integer('model_cant_fc1', 250, 'Cantidad de neurolas full conected en capa 3')

