#RECONOBOOK
###Un proyecto open source de visión artificial para reconocer la portada de libros.

1 - Objetivos del proyecto
2 - Herramientas utilizadas
3 - Descripción del problema
4 - Obtención de imagenes
5 - Generación de datasets a partir de imagenes
6 - Armado del modelo, descripción de la red neuronal
7 - Entrenamiento del modelo
8 - Evaluación del modelo
9 - Conclusiones
10 - Lista TO DOs
    [] Correr en 2 CPU
    [] Correr en GPU y CPU
    [] Correr el modelo en 2 servidores distribuidos
    [] Utilizar el modelo para construir un programa que evalue fotos
    [] Integrar el programa que evalua fotos a una camara web y evaluar "online"
    [] Agregar al tensorboard las visualización de las imagenes que se predicen incorrectamente
    [] Correr el modelo en Android
    [] Ampliar a 30 Libros


Extras:
    Indice del repositorio
    Script instalación de tensorflow en UBUNTU 64
    IDE utilizado
    Programa para separar frames en .jpg a partir de un video
    Atajos rapidos desde consola
    Links de interes


-----------------------------------------------------------

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

**Archivos vinculados a la configuración general:**

- **config.py ->** Contiene todas las constantes parametrizables del sistema (FLAGS).

**Archivos de modelos y datasets viejos:**

- **Resguardo/* ->** Contiene modelos y datasets que no son el corriente.

-----------------------------------------------------------


###Paso a paso, desde imagenes a modelo entrenado:

0. **Crear carpetas necesarias**
  - Crear las siguientes carpetas:
    - /summary_eval
    - /summary_train
    - /datasets
    - /checkpoints
1. **Cargar imagenes en la carpeta /imagenes_jpg**
  - Con una subcarpeta por cada caterogia.
2. **Run dataset_scripts/split_dataset.py**,
  - Parametro Modificable: porcentaje_img_test
  - Al finalizar se crean carpetas de train y test y el archivo con los labels.
3. **Run dataset_scripts/build_datasets.py**
  - Parametro modificable: porcentaje_img_test
  - Al finalizar se crean los datasets de train y validation en /datasets
4. **Configurar Dataset**:
  - ir a config.py y completar cantidad_imagenes_train y cantidad_imagenes_eval de acuerdo a las imagenes en las carpetas
5. **Run reconobook_train.py**
  - Hay algunos parametros que se pueden modificar en config.py, revisar.
6. **Run reconobook_eval.py**
  - Parametros modificable: config.py->eval_unique, indica si se evaluan todas imagenes juntas o una por vez.


-----------------------------------------------------------


### Utils:

**Abrir IDE**
```
./Downloads/pycharm-community-2016.1.2/bin/pycharm.sh
```

**Visualizar TensorBoard**
```
tensorboard --logdir=/home/presta/ReconoBook/summary_train

o

tensorboard --logdir=/home/presta/ReconoBook/summary_eval
```

**Instalación TensorFlow en UBUNTU 64**
```
sudo apt-get install python-pip python-dev
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl
export LC_ALL=C
sudo pip install --upgrade $TF_BINARY_URL
git clone https://github.com/NicolasPresta/ReconoBook.git
cd ReconoBook/
mkdir summary_eval
mkdir summary_train
mkdir datasets
mkdir checkpoints
cd dataset_scripts/
python build_datasets.py
cd ..
python reconobook_train.py

...

```
-----------------------------------------------------------







