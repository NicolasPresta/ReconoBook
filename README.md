#RECONOBOOK

##Un proyecto open source de visión artificial para reconocer la portada de libros.

![Portada](./doc/img/portada2.jpg "Portada")

***

###INDICE
1.  [Introducción y Objetivos](./doc/objetivos.md)
2.  [Herramientas utilizadas](./doc/herramientas.md)
3.  [Descripción del problema](./doc/problema.md)
4.  [Obtención de imagenes](./doc/obtencion.md)
5.  [Generación de datasets](./doc/generacion.md)
6.  Armado del modelo, descripción de la red neuronal
7.  Entrenamiento del modelo
8.  Evaluación del modelo
9.  Conclusiones
10. [Lista TODOs](./doc/TODO.md)


####Extras:
- [Indice del repositorio](./doc/indicerepo.md)
- [Script instalación de Tensorflow en UBUNTU 64](./doc/ubuntu.md)
- [IDE utilizado](https://www.jetbrains.com/pycharm/)
- [Programa para separar frames en .jpg a partir de un video](https://www.dvdvideosoft.com/es/products/dvd/Free-Video-to-JPG-Converter.htm)
- [Atajos rapidos desde consola](./doc/atajos.md)
- [Links de interes](./doc/links.md)


***

Sobre el Autor:

**Nicolás Rodriguez Presta**


**Desarollador** fullstack  
**Estudiante** de Ingenieria en Sistemas   
**Entusiasta** de Machine Learning  


[LinkedIn](https://www.linkedin.com/in/nicolaspresta/) / 
[Twitter](https://twitter.com/nicolaspresta)

***


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


***




