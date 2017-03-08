## Obtención de imagenes

Para *entrenar* el modelo contamos con varias imagenes de las portadas de cada libro.  

Para **obtener las imagenes se grabó un pequeño video** (10 seg) posicionando la portada entrente de la camara.  

El libro se fue moviendo e inclinando, alejando y acercando, para **cubrir la mayor cantidad de posiciones posibles** 
(El modelo debe ser capas de reconocer el libro en cualquiera de estas posiciones), esto tambien ayuda a prevenir el 
sobreajuste.  

Se realizaron varias capturas en video y su posterior separacion de frames en imagenes. Esto es para prevenir que las 
condiciones de luz puedan sobreajustar el modelo. **En total se realizaron 3 capuras (A, B y C)**.  
Para evitar que haya un sesgo a clasificar mejor las imagenes de una determinada captura, **se utiliza la misma cantidad
de imagenes para cada captura (y cada libro)**.  

Cada libro tiene en total 330 imagenes, **110 corresponden a la captura A, 110 a la B y 110 a la C**.  
**Las 110 imagenes de cada captura se eligieron aleatoriamente**. 

Las imagenes se separan por carpeta. El formato de las imagenes es *jpg* y el tamaño puede variar, pero en general es de
640x480 (de todos modos el tamaño no influye, ya que para alimentar el modelo se van a redimensionar a una escala menor)

Las imagenes se encuentran en la carpeta */imagenes_jpg*, El nombre de cada subcarpeta es el ID del libro.  
El nombre de la imagen tiene el siguiente formato: <ID-Libro><ID-Capura> <Nro imagen>.jpg  
Por ejemplo la imagen '**1A 005.jpg**' corresponde al libro ID 1, captura A.  

Contamos como herramienta con un script que analisa el conjunto de imagenes para totalizar cuantas imagenes hay 
por libro y captura.  
Para esto se utiliza el script [analize_jpg.py](../dataset_scripts/analize_jpg.py)

![Obtencion](./img/obtencion1.png "Obtencion")  


```shell
python analize_jpg.py
```

![Obtencion](./img/obtencion2.png "Obtencion")


[<- Volver Home](../README.md)