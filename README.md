#RECONOBOOK
###Un proyecto open source de visión artificial para reconocer la portada de libros.

**Conocimientos aplicados:** Machine Learning, Deep Learning, Convolutional neural networks.

**Herramientras aplicadas:** TensorFlow by Google, Python.

-----------------------------------------------------------

###Indice del repositorio:

**Archivos vinculados al armado y evaluación del modelo:**

- **base_dataset.py ->** Clase base de un dataset. No tiene mayor importancia.
- **reconobook_dataset.py ->** Clase que representa el dataset de Reconobook. Hereda de dataset. No tiene mayor ciencia.
- **reconobook_eval.py ->** Contiene métodos para evalular un modelo.
  - evaluate: Evalua un modelo contra el dataset de testing completo
  - evaluate_unique: Evalua el modelo una imagen a la vez, mostrandola por pantalla.
- **reconobook_input.py ->** Contiene los metodos que sirven de input al modelo, leyendo de los datasets.
- **reconobook_modelo.py ->** El corazón del proyecto, acá se define y arma el modelo.
  - inference: dada una imagen retorna su preducción
  - loss: calcula la perdida de una predicción.
  - train: backpropagation! (GradientDescentOptimizer)
- **reconobook_train.py ->** Orquesta el entrenamiento del modelo

**Archivos vinculados al dataset (procesar los .jpg y armar un dataset):**

- **imagenes_jpg/* ->** Imagenes jpg en crudo organizadas por clases.
- **dataset_scripts/split_dataset.py ->** divide el conjunto de imagenes en 2 conjuntos: uno de test y otro de evaluación.
- **dataset_scripts/build_dataset.py ->** luego de divididas las imagenes en 2 carpetas, crea 2 datasets. (jpg -> dataset).
- **datasets/* ->** Datasets armados y listos para ser pasados al modelo!

**Archivos vinculados al modelo ya entrenado:**

- **checkpoints/* ->** Modelo ya entrenado, listo para ser evaluado.


-----------------------------------------------------------


###Paso a paso, desde imagenes a modelo entrenado:

1. **Cargar imagenes en la carpeta /imagenes_jpg**
  - Con una subcarpeta por cada caterogia.
2. **Run dataset_scripts/split_dataset.py**,
  - Completar labels.txt
  - Parametro Modificable: porcentaje_img_test
  - Al finalizar se crean carpetas de train y test.
3. **Run dataset_build_datasets.py**
  - Parametro modificable: porcentaje_img_test
  - Al finalizar se crean los datasets de train y validation en /datasets
4. **Configurar Dataset**:
  - NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = (cantidad de imagenes de entrenamiento)
  - NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = (cantidad de imagenes para evaluar)
  - NUM_CLASES = (cantidad de clases)
5. **Run reconobook_train.py**
  - Hay algunos parametros que se pueden modificar, revisar.
6. **Run reconobook_eval.py**
  - Parametros modificable: unique, indica si se evaluan todas imagenes juntas o una por vez.



-----------------------------------------------------------






