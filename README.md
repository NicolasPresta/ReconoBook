#RECONOBOOK
##Un proyecto open source de visión artificial para reconocer la portada de libros.

**Conocimientos aplicados:** Machine Learning, Deep Learning, Convolutional neural networks.

**Herramientras aplicadas:** TensorFlow by Google, Python.


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








