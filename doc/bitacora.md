
[<- Volver Home](../README.md)  

## Bitacora de progreso
Resumen de los cambios que se van haciendo al proyecto.

***  

**09/03/2017**

Luego de 70000 pasos se obtiene un modelo entrenado con esta precición para top_k = 1:

- TRAIN:  precision = 0.864
- VALIDATION:  precision = 0.844
- TEST:  precision = 0.628  

Vemos que el modelo no lo está haciendo bien para las imagenes de la captura F. El modelo sufre de alta varianza, no está generaizando bien. 


*** 

**15/03/2017**

Se cambia la estructura del modelo agregando más kernels y se reentrena:

model_cant_kernels1 = 60  
model_cant_kernels2 = 120  
model_cant_fc1 = 250  

![Bitacora](./img/bitacora1b.JPG "Bitacora")
![Bitacora](./img/bitacora1c.JPG "Bitacora")
![Bitacora](./img/bicatora1a.JPG "Bitacora")


Vemos que el modelo lo hace muy bien con los sets de test y validacion, pero mal con el de test. Osea que no puede generalizar el aprendizaje para evaluar imagenes de otras capturas correctamente. Es decir que el modelo está sobreajustado. 

***

**16/03/2017**

Se cambia la estructura, disminuyendo la cantidad de kernels a la mitad y se reentrena

model_cant_kernels1 = 30  
model_cant_kernels2 = 60  
model_cant_fc1 = 125  

![Bitacora](./img/bitacora2b.PNG "Bitacora")
![Bitacora](./img/bitacora2c.PNG "Bitacora")
![Bitacora](./img/bicatora2a.PNG "Bitacora")  

Vemos que el modelo lo hace muy bien con los sets de test y validacion, pero mal con el de test, aunque mejoró respecto de la estructura anterior, ademas de que mejoró el tiempo de entrenamiento.  
En el paso 10000 se obtiene la mejor precisión, luego comienza a disminuir producto del sobreajuste.  
Aun el modelo no lo hace como quisieramos para el set de test, osea para las capturas nuevas.  


*** 

**17/03/2017**

La estructura del modelo es igual a la del punto anterior, pero se agrega un dropout de la capa FC1, con un 50% de probabilidades de que se activen o no las neuronas (buscamos regularizar el modelo para que generalice mejor)

Los resultados son estos:

![Bitacora](./img/bitacora3b.JPG "Bitacora")
![Bitacora](./img/bitacora3c.JPG "Bitacora")
![Bitacora](./img/bitacora3a.JPG "Bitacora")  

Vemos que no hay mejoras en la captura de test, llega hasta un 65% de acierto, frente al 99% del set de entrenamiento y 95% del set de validación.

***

**20/03/2017**

Usando la misma estructura, se agrega regularización a los parametros de los kernels de convolución. 
wd=0.0004 (weight decay)

Se entrena el modeo por 42900 pasos y se evalua:

![Bitacora](./img/bitacora4b.JPG "Bitacora")
![Bitacora](./img/bitacora4c.JPG "Bitacora")
![Bitacora](./img/bitacora4a.JPG "Bitacora")  

Vemos que el modelo mejora bastante en el set de test. La regularización es un buen camino. 
En los sets de train y validation la precición es muy elevada (> 98%) y en el set de test es cercana al 75%. 
El modelo está aprendiendo mejor. 

***
[<- Volver Home](../README.md)  
