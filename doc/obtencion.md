[<- Volver Home](../README.md)

## Obtención de imagenes

<img src="./img/1C.jpg" width="80">
<img src="./img/2C.jpg" width="80">
<img src="./img/3C.jpg" width="80">
<img src="./img/4C.jpg" width="80">
<img src="./img/5C.jpg" width="80"> 
<img src="./img/6C.jpg" width="80"> 
<img src="./img/7C.jpg" width="80"> 
<img src="./img/8C.jpg" width="80"> 
<img src="./img/9C.jpg" width="80"> 
<img src="./img/10C.jpg" width="80"> 
<img src="./img/11C.jpg" width="80"> 
<img src="./img/12C.jpg" width="80">
<img src="./img/13C.jpg" width="80"> 
<img src="./img/14C.jpg" width="80"> 
<img src="./img/15C.jpg" width="80"> 
<img src="./img/16C.jpg" width="80"> 
<img src="./img/17C.jpg" width="80"> 
<img src="./img/18C.jpg" width="80"> 
<img src="./img/19C.jpg" width="80"> 
<img src="./img/20C.jpg" width="80">  


Para *entrenar y validar* el modelo contamos con varias imagenes de las portadas de cada libro.  

Para **obtener las imagenes se grabó un pequeño video** (10 seg) posicionando la portada frente de la camara.  

El libro se fue moviendo e inclinando, alejando y acercando, para **cubrir la mayor cantidad de posiciones posibles** 
(El modelo debe ser capas de reconocer el libro en cualquiera de estas posiciones), esto tambien ayuda a prevenir el 
sobreajuste.  

Se realizaron varias capturas en video y su posterior separacion de frames en imagenes. Esto es para prevenir que las 
condiciones de luz y del fondo puedan sobreajustar el modelo y hacer que no generalice bien el aprendizaje. **En total se realizaron 6 capuras para entrenar, validar y testear el modelo (A, B, C, D, E y F)**.
Para evitar que haya un sesgo a clasificar mejor las imagenes de una determinada captura, **se utiliza la misma cantidad
de imagenes para cada captura (y cada libro)**.  

Cada libro tiene en total 660 imagenes, **110 corresponden a la captura A, 110 a la B y 110 a la C, etc**.  
**Las 110 imagenes de cada captura se eligieron aleatoriamente**. 

Las imagenes se separan por carpeta. El formato de las imagenes es *jpg* y el tamaño puede variar, pero en general es de
640x480 (de todos modos el tamaño no influye, ya que para alimentar el modelo se van a redimensionar a una escala menor)

Las imagenes se encuentran en la carpeta */imagenes_jpg*, El nombre de cada subcarpeta es el ID del libro. (Todas las
imagenes se subieron al repositorio de github para faciliar su obtención y descarga)
El nombre de la imagen tiene el siguiente formato: <ID-Libro><ID-Capura> <Nro imagen>.jpg  
Por ejemplo la imagen '**1A 005.jpg**' corresponde al libro ID 1, captura A.  

Contamos como herramienta con un script que analisa el conjunto de imagenes para totalizar cuantas imagenes hay 
por libro y captura.  
Para esto se utiliza el script [analize_jpg.py](../dataset_scripts/analize_jpg.py)

![Obtencion](./img/obtencion1.png "Obtencion")  

```shell
cd dataset_scripts/
python analize_jpg.py
```

![Obtencion](./img/obtencion2.png "Obtencion")


### Libros a clasificar:

1. Fisica universita  
A)<img src="./img/1A.jpg" width="120"> B)<img src="./img/1B.jpg" width="120"> C)<img src="./img/1C.jpg" width="120"> D)<img src="./img/1D.jpg" width="120">  E)<img src="./img/1E.jpg" width="120">  F)<img src="./img/1F.jpg" width="120">  

2. Patrones de diseño  
A)<img src="./img/2A.jpg" width="120"> B)<img src="./img/2B.jpg" width="120"> C)<img src="./img/2C.jpg" width="120"> D)<img src="./img/2D.jpg" width="120">  E)<img src="./img/2E.jpg" width="120">  F)<img src="./img/2F.jpg" width="120">  

3. Introducción a Mineria de datos  
A)<img src="./img/3A.jpg" width="120"> B)<img src="./img/3B.jpg" width="120"> C)<img src="./img/3C.jpg" width="120"> D)<img src="./img/3D.jpg" width="120">  E)<img src="./img/3E.jpg" width="120">  F)<img src="./img/3F.jpg" width="120">  

4. Mineria de datos a traves de ejemplos  
A)<img src="./img/4A.jpg" width="120"> B)<img src="./img/4B.jpg" width="120"> C)<img src="./img/4C.jpg" width="120"> D)<img src="./img/4D.jpg" width="120">  E)<img src="./img/4E.jpg" width="120">  F)<img src="./img/4F.jpg" width="120">  

5. Sistemas expertos  
A)<img src="./img/5A.jpg" width="120"> B)<img src="./img/5B.jpg" width="120"> C)<img src="./img/5C.jpg" width="120"> D)<img src="./img/5D.jpg" width="120">  E)<img src="./img/5E.jpg" width="120">  F)<img src="./img/5F.jpg" width="120">  

6. Sistemas inteligentes  
A)<img src="./img/6A.jpg" width="120"> B)<img src="./img/6B.jpg" width="120"> C)<img src="./img/6C.jpg" width="120"> D)<img src="./img/6D.jpg" width="120">  E)<img src="./img/6E.jpg" width="120">  F)<img src="./img/6F.jpg" width="120">  

7. Big data  
A)<img src="./img/7A.jpg" width="120"> B)<img src="./img/7B.jpg" width="120"> C)<img src="./img/7C.jpg" width="120"> D)<img src="./img/7D.jpg" width="120">  E)<img src="./img/7E.jpg" width="120">  F)<img src="./img/7F.jpg" width="120">  

8.  Analisis matematico (vol 3 / Azul)  
A)<img src="./img/8A.jpg" width="120"> B)<img src="./img/8B.jpg" width="120"> C)<img src="./img/8C.jpg" width="120"> D)<img src="./img/8D.jpg" width="120">  E)<img src="./img/8E.jpg" width="120">  F)<img src="./img/8F.jpg" width="120">  

9.  Einstein  
A)<img src="./img/9A.jpg" width="120"> B)<img src="./img/9B.jpg" width="120"> C)<img src="./img/9C.jpg" width="120"> D)<img src="./img/9D.jpg" width="120">  E)<img src="./img/9E.jpg" width="120">  F)<img src="./img/9F.jpg" width="120">  

10. Analisis matematico (vol 2 / Amarillo)  
A)<img src="./img/10A.jpg" width="120"> B)<img src="./img/10B.jpg" width="120"> C)<img src="./img/10C.jpg" width="120"> D)<img src="./img/10D.jpg" width="120">  E)<img src="./img/10E.jpg" width="120">  F)<img src="./img/10F.jpg" width="120">  

11. Teoria de control  
A)<img src="./img/11A.jpg" width="120"> B)<img src="./img/11B.jpg" width="120"> C)<img src="./img/11C.jpg" width="120"> D)<img src="./img/11D.jpg" width="120">  E)<img src="./img/11E.jpg" width="120">  F)<img src="./img/11F.jpg" width="120">  

12. Empresas de consultoría  
A)<img src="./img/12A.jpg" width="120"> B)<img src="./img/12B.jpg" width="120"> C)<img src="./img/12C.jpg" width="120"> D)<img src="./img/12D.jpg" width="120">  E)<img src="./img/12E.jpg" width="120">  F)<img src="./img/12F.jpg" width="120">  

13. Legislación  
A)<img src="./img/13A.jpg" width="120"> B)<img src="./img/13B.jpg" width="120"> C)<img src="./img/13C.jpg" width="120"> D)<img src="./img/13D.jpg" width="120">  E)<img src="./img/13E.jpg" width="120">  F)<img src="./img/13F.jpg" width="120">  

14. En cambio  
A)<img src="./img/14A.jpg" width="120"> B)<img src="./img/14B.jpg" width="120"> C)<img src="./img/14C.jpg" width="120"> D)<img src="./img/14D.jpg" width="120">  E)<img src="./img/14E.jpg" width="120">  F)<img src="./img/14F.jpg" width="120">  

15. Liderazgo Guardiola  
A)<img src="./img/15A.jpg" width="120"> B)<img src="./img/15B.jpg" width="120"> C)<img src="./img/15C.jpg" width="120"> D)<img src="./img/15D.jpg" width="120">  E)<img src="./img/15E.jpg" width="120">  F)<img src="./img/15F.jpg" width="120">  

16. Constitución Argentina  
A)<img src="./img/16A.jpg" width="120"> B)<img src="./img/16B.jpg" width="120"> C)<img src="./img/16C.jpg" width="120"> D)<img src="./img/16D.jpg" width="120">  E)<img src="./img/16E.jpg" width="120">  F)<img src="./img/16F.jpg" width="120">  

17. El arte de conversar  
A)<img src="./img/17A.jpg" width="120"> B)<img src="./img/17B.jpg" width="120"> C)<img src="./img/17C.jpg" width="120"> D)<img src="./img/17D.jpg" width="120">  E)<img src="./img/17E.jpg" width="120">  F)<img src="./img/17F.jpg" width="120">  

18. El señor de las moscas  
A)<img src="./img/18A.jpg" width="120"> B)<img src="./img/18B.jpg" width="120"> C)<img src="./img/18C.jpg" width="120"> D)<img src="./img/18D.jpg" width="120">  E)<img src="./img/18E.jpg" width="120">  F)<img src="./img/18F.jpg" width="120">   

19. Revista: Epigenetica  
A)<img src="./img/19A.jpg" width="120"> B)<img src="./img/19B.jpg" width="120"> C)<img src="./img/19C.jpg" width="120"> D)<img src="./img/19D.jpg" width="120">  E)<img src="./img/19E.jpg" width="120">  F)<img src="./img/19F.jpg" width="120">  

20. Revista: Lado oscuro del cosmos  
A)<img src="./img/20A.jpg" width="120"> B)<img src="./img/20B.jpg" width="120"> C)<img src="./img/20C.jpg" width="120"> D)<img src="./img/20D.jpg" width="120">  E)<img src="./img/20E.jpg" width="120">  F)<img src="./img/20F.jpg" width="120">  


***
[<- Volver Home](../README.md)
