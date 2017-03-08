##Paso a paso, desde imagenes a modelo entrenado:

Este es un breve resumen de los pasos para obtener el modelo entrenado y evaluarlo.

1. **Cargar imagenes en la carpeta /imagenes_jpg**
  - Con una subcarpeta por cada caterogia.
2. **Run dataset_scripts/split_dataset.py**,
  - Parametro Modificable: porcentaje_img_test
  - Al finalizar se crean carpetas de train y test y el archivo con los labels.
3. **Run dataset_scripts/build_datasets.py**
  - Al finalizar se crean los datasets de train y validation en /datasets
4. **Configurar Dataset**:
  - ir a config.py y completar cantidad_imagenes_train y cantidad_imagenes_eval de acuerdo a las imagenes en las carpetas
5. **Run reconobook_train.py**
  - Hay varios parámetros que se pueden modificar en config.py.
6. **Run reconobook_eval.py**
  - Parametros modificable: config.py->eval_unique, indica si se evaluan todas imagenes juntas o una por vez.