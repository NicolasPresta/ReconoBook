##Indice del repositorio:

####Datos (imagenes) para entrenar y evaluar el modelo:

- imagenes_jpg/* : En esta carpeta se encuentran todas las imagenes jpg de las portadas de libros. 
                   Cada subcarpeta representa una clase (un libro) distinto.
                   el nombre del archivo de la imagen se forma de la siguiente forma: '<ID-Libro><ID-Captura> <Nro imagen>.jpg'
                   
- manual_test_img/* : En esta carpeta se pueden colocar imagenes de las portadas de libros tomadas por fuera de las capturas para evaluarlas una a una. Todas las imagenes que se encuentren en esta carpeta pueden ser usadas como input del modelo ya entrenado para que las clasifique

####Scripts vinculados al dataset (procesar los .jpg y armar un dataset):

- dataset_scripts/
    - *analize_jpg.py*: Analiza las imagenes que se encuentran en /imagenes_jpg.
                          Genera un informe sobre la cantidad de imagenes para cada libro y captura. 
    - *split_dataset.py*: divide el conjunto de imagenes en 2 conjuntos: uno de test y otro de evaluación.
                            Genera una nueva carpeta con los 2 subconjutos de imagenes. 
    - *build_dataset.py*: A partir de las carpetas generadas por split_dataset.py, genera 2 datasets de registros TFRecords.

####Documentación:

- doc/* : En esta carpeta se encuentra toda la documentación del proyecto.


####Modelo, entrenamiento y evaluación:
- *reconobook_dataset.py*: Clase que representa el dataset de Reconobook. Puede ser un dataset de test o de entrenamiento.
- *reconobook_input.py*: Contiene los metodos que sirven de input al modelo, leyendo de los datasets.
- *reconobook_modelo.py:* El corazón del proyecto, acá se define y arma el modelo.
- *reconobook_train.py:* Orquesta el entrenamiento del modelo
- *reconobook_eval.py:* Contiene métodos para evalular un modelo.

####Archivos vinculados al modelo ya entrenado

- checkpoints/*: Modelo ya entrenado, listo para ser evaluado. Cada 1000 pasos de entrenamiento se guarda un snapshot del modelo.

####Configuración general

- *config.py*: Contiene todas las constantes parametrizables del sistema (FLAGS).

[<- Volver Home](../README.md)
