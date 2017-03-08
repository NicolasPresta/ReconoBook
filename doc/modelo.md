[<- Volver Home](../README.md)  


## Descripcón del modelo


El input de nuestor modelo serán imagenes de 40x40 px. Por lo tanto es necesario redimensionar las 
imagenes antes de alimentar el modelo.  
Las imagenes son RGB, por lo tanto cada imagen es un vector de [40,40,3].  
Para el entrenamiento se van a aplicar transformaciones en las imagenes. Esto permite aumentar artificialmente la cantidad de imagenes distintas.  
El **preprocesado** de cada imagen incluye:
- Alterar aleatoriamente el contraste.
- Alterar aleatoriamente el matiz.
- Alterar aleatoriamente la saturación.
- Recortar aleatoriamente un porcion de la imagen de 40x40 que represente el 70% y el 100% de la imagen original.

Todo el tratamiento de las imagenes para alimentar el modelo está en *reconobook_input.py*. Tensorflow nos permite tener varios hilos realizando la lectura y el preprocesado en simultáneo.

![Modelo](./img/modelo3.jpg "Modelo")

La red neuronal tendrá la siguiente arquitectura:

![Modelo](./img/modelo4.jpg "Modelo")

Podemos ver el grafo de ejecución que genera TensorFlow para darnos una mejor idea de lo que en realidad está ocurriendo

![Modelo](./img/modelo1.jpg "Modelo")

Donde el Input es la caja de abajo llamada *batch_processing*, podemos hacer zoom en esta parte para visualizar los 2 hilos de ejecución que se encargan de leer el dataset, parsear el TFRecord, decodificar el jpg y preprocesar la imagen. Luego se unen en *batch_join* para ser de input a la primer capa de la red neuronal.

![Modelo](./img/modelo2.jpg "Modelo")



***
[<- Volver Home](../README.md)  
