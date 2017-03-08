## Generación de datasets

Las imagenes que tenemos en */imagenes_jpg*  son todas* las que tenemos, por lo tanto las vamos a utilizar tanto para 
entrenar como para evaluar el modelo.  
(*Adicionalmente se pueden incluir imagenes en la carpeta */manual_test_img* para poder inyectarselas al modelo 
ya entrenado para que las evalue)

Por eso primero necesitamos **dividir las imagenes en dos sets**, un conjunto de entrenamiento y otro conjunto de 
evaluación.  
Para esto se utiliza el script [split_dataset.py](../dataset_scripts/split_dataset.py)
 
Una vez con las imagenes divididas en 2 conjuntos necesitamos armar los **Archivos de dataset**, ya que tensorflow
no leerá directamente las imagenes, sino que irá recorriendo un archivo que tiene todas las imagenes serializadas.
El archivo de dataset es un archivo de TFRecords. Para más información [ver la documentación de tensorflow]
(https://www.tensorflow.org/api_guides/python/python_io#tfrecords_format_details)  
Para esto se utiliza el script [build_dataset.py](../dataset_scripts/build_dataset.py)


Diagrama del flujo para obtener el dataset separado y formateado:


![Obtencion](./img/generacion1.png "Obtencion")
