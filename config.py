# coding=utf-8

# ==============================================================================

import tensorflow as tf

# ==============================================================================

FLAGS = tf.app.flags.FLAGS

# ==============================================================================

# -------------------------------- DIRECTORIOS --------------------------------
tf.app.flags.DEFINE_string('summary_dir_eval', './summary_eval', "Logs de proceso de evaluación")
tf.app.flags.DEFINE_string('summary_dir_train', './summary_train', "Logs de proceso de entrenamiento")
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints', "Resguardo del modelo a utilizar")
tf.app.flags.DEFINE_string('datasets_dir', './datasets', 'Directorio donde esttán las imagenes a utilizar')
tf.app.flags.DEFINE_string('img_dir', './imagenes_jpg', 'Directorio donde esttán las imagenes a utilizar')
tf.app.flags.DEFINE_string('manual_test_folder', './manual_test_img/', 'Directorio con imagenes a evaluar manualmente')
tf.app.flags.DEFINE_string('data_split_dir', './split_jpg', 'Directorio donde estan las imagenes divididas a utilizar')

tf.app.flags.DEFINE_string('train_folder', 'train', 'Nombre directorio con las imagenes de entrenamiento')
tf.app.flags.DEFINE_string('validation_folder', 'validation', 'Nombre Directorio con las imagenes de validation')
tf.app.flags.DEFINE_string('test_folder', 'test', 'Nombre Directorio con las imagenes de test')

tf.app.flags.DEFINE_string('labels_file_name', 'labels.txt', 'Labels file')

tf.app.flags.DEFINE_string('export_model_dir', './export_model/', 'Carpeta donde se exporta el modelo')

# -------------------------------- DATASET --------------------------------
tf.app.flags.DEFINE_string('porcentaje_img_validation', 20, 'Porcentaje de imagenes que van al set de validation')
tf.app.flags.DEFINE_integer('cantidad_clases', 20, 'Cantidad de clases a reconocer')

tf.app.flags.DEFINE_boolean('modo_procesar', False, 'Al ejecutar analize_jpg realizar emparejamiento')
tf.app.flags.DEFINE_integer('img_por_captura', 110, 'Cantidad de imagenes a conservar por captura')

tf.app.flags.DEFINE_string('capturas_id', 'A,B,C,D,E,F', 'Ids de todas las capturas realizadas')
tf.app.flags.DEFINE_string('capturasEntrenamiento_id', 'A,B,C,D,E', 'Ids de todas las capturas para entrenar')
tf.app.flags.DEFINE_string('capturasTest_id', 'F', 'Ids de todas las capturas para test')

tf.app.flags.DEFINE_integer('train_shards', 1, 'Numero de particiones del dataset de entrenamiento')
tf.app.flags.DEFINE_integer('validation_shards', 1, 'Numero de particiones del dataset de validación')
tf.app.flags.DEFINE_integer('test_shards', 1, 'Numero de particiones del dataset de entrenamiento')
tf.app.flags.DEFINE_integer('dataset_num_threads', 1, 'Numero de hilos de ejecución para armar el dataset')


# -------------------------------- INPUT --------------------------------
tf.app.flags.DEFINE_integer('input_num_preprocess_threads', 2, 'Numero de hilos que hacen el preprocesado')
tf.app.flags.DEFINE_integer('input_num_readers', 2, 'Numero de readers')
tf.app.flags.DEFINE_integer('image_height', 40, 'Alto imagen')
tf.app.flags.DEFINE_integer('image_width', 40, 'Ancho imagen')


# -------------------------------- MODELO --------------------------------
tf.app.flags.DEFINE_integer('model_cant_kernels1', 30, 'Cantidad de kernels de convolución en la capa 1')
tf.app.flags.DEFINE_integer('model_cant_kernels2', 60, 'Cantidad de kernels de convolución en la capa 2')
tf.app.flags.DEFINE_integer('model_cant_fc1', 125, 'Cantidad de neurolas full conected en capa 3')

tf.app.flags.DEFINE_integer('model_version', 1, 'Versión del modelo')


# -------------------------------- ENTRENAMIENTO --------------------------------
tf.app.flags.DEFINE_integer("moving_average_decay", 0.9999, "The decay to use for the moving average.")
tf.app.flags.DEFINE_integer("initial_learning_rate", 0.09, "Initial learning rate.")
tf.app.flags.DEFINE_integer("decay_steps", 1000, "Epochs after which learning rate decays.")
tf.app.flags.DEFINE_integer("decay_rate", 0.95, "Learning rate decay factor.")

tf.app.flags.DEFINE_boolean('log_device_placement', False, "Si logea la ubicación de variables al inciar la ejecución")
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, "Si permite una asignación de variables flexible")

tf.app.flags.DEFINE_boolean('train_distort', True, "Distorcionar imagenes al evaluar")
tf.app.flags.DEFINE_boolean('train_crop', True, "Distorcionar imagenes al evaluar")

tf.app.flags.DEFINE_integer('train_max_steps', 1000000, "Number of batches to run.")
tf.app.flags.DEFINE_integer("train_batch_size", 128, "Cantidad de imagenes que se procesan por batch")

tf.app.flags.DEFINE_integer("steps_to_imprimir_avance", 50, "Cantidad de pasos cada los cuales se imprimer por consola")
tf.app.flags.DEFINE_integer("steps_to_guardar_summary", 100, "Cantidad de pasos cada los cuales se guarda summary")
tf.app.flags.DEFINE_integer("steps_to_guardar_checkpoint", 500, "Cantidad de pasos cada los cuales se guarda checkpoint")
tf.app.flags.DEFINE_integer("saver_max_to_keep", 100, "Cantidad de checkouts a concervar")

tf.app.flags.DEFINE_boolean('use_dropout_1', False, "Si usa drop out en capa 1")
tf.app.flags.DEFINE_boolean('use_dropout_2', False, "Si usa drop out en capa 2")
tf.app.flags.DEFINE_boolean('use_dropout_3', False, "Si usa drop out en capa 3")
tf.app.flags.DEFINE_boolean('use_dropout_4', False, "Si usa drop out en capa 4")
tf.app.flags.DEFINE_integer("keep_drop_prob", 0.5, "probabilidad de quedar en el drop out")

tf.app.flags.DEFINE_string('optimezer', 'GradientDescentOptimizer', 'Optimizador a usar: '
                                                                    'GradientDescentOptimizer, '
                                                                    'AdamOptimizer, '
                                                                    'AdadeltaOptimizer, '
                                                                    'RMSPropOptimizer, '
                                                                    'ProximalGradientDescentOptimizer')


# -------------------------------- EVALUACION --------------------------------
tf.app.flags.DEFINE_boolean('eval_unique', False, "Ejecutar revisión imagen por imagen")
tf.app.flags.DEFINE_boolean('eval_unique_from_dataset', True, "Evaluar imagen por imagen desde dataset")
tf.app.flags.DEFINE_integer('eval_unique_cantidad_img', 3, "Cantidad de imagenes a evaluar si eval_unique = true")
tf.app.flags.DEFINE_string('eval_dataset', 'test', 'Data set usado para validacion (train, validation o test')

tf.app.flags.DEFINE_boolean('eval_distort', False, "Distorcionar imagenes al evaluar")
tf.app.flags.DEFINE_boolean('eval_crop', True, "Distorcionar imagenes al evaluar")

tf.app.flags.DEFINE_integer('eval_num_examples', 2200, "Número de imagenes a evaluar")
tf.app.flags.DEFINE_integer('eval_num_examples_mini', 1000, "Número de imagenes a evaluar durante el entrenamiento")

tf.app.flags.DEFINE_integer("top_k_prediction", 1, "La predicción correcta si esta entre los k primeros resultados")

tf.app.flags.DEFINE_boolean('visualice_conv1_kernels', False, "Hacer Summary de kernels")

titulosStr = ("Fisica universita,"
              "Patrones de diseño,"
              "Introducción a Mineria de datos,"
              "Mineria de datos a traves de ejemplos,"
              "Sistemas expertos,"
              "Sistemas inteligentes,"
              "Big data,"
              "Analisis matematico (vol 3 / Azul),"
              "Einstein,"
              "Analisis matematico (vol 2 / Amarillo),"
              "Teoria de control,"
              "Empresas de consultoría,"
              "Legislación,"
              "En cambio,"
              "Liderazgo Guardiola,"
              "Constitución Argentina,"
              "El arte de conversar,"
              "El señor de las moscas,"
              "Revista: Epigenetica,"
              "Revista: Lado oscuro del cosmos")

tf.app.flags.DEFINE_string('titulos', titulosStr, 'Titulos de los libros')







