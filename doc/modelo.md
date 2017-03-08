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

![modelo](../README.md)  



***
[<- Volver Home](../README.md)  
