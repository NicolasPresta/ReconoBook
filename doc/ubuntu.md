[<- Volver Home](../README.md)  

##Instalación TensorFlow en UBUNTU 64

Para tener una mejor idea sobre la instalación es recomendable [ver la documentación desde la página de TensorFlow](https://www.tensorflow.org/install/)

###Instalar tensorflow


```shell
> sudo apt-get install python-pip python-dev
> export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl
> export LC_ALL=C
> sudo pip install --upgrade $TF_BINARY_URL

```

###Obtener repositorio, separar imagenes en train y test, armar datasets, entrenar modelo:


```shell
> git clone https://github.com/NicolasPresta/ReconoBook.git
> cd ReconoBook/
> cd dataset_scripts/
> python split_dataset.py
> python build_datasets.py
> cd ..
> python reconobook_train.py
```


  ***
[<- Volver Home](../README.md)
