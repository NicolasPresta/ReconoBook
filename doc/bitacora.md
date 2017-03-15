
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

![Bitacora](./img/bitacora1a.JPG "Bitacora")
![Bitacora](./img/bitacora1b.JPG "Bitacora")
![Bitacora](./img/bitacora1c.JPG "Bitacora")

Vemos que el modelo lo hace muy bien con los sets de test y validacion, pero mal con el de test. Osea que no puede generalizar el aprendizaje para evaluar imagenes de otras capturas correctamente. Es decir que el modelo está sobreajustado. 



*** 
[<- Volver Home](../README.md)  
