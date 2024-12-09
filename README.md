# Clasificación enfermedades de la retina

### Alicia García Espiga

## Contenido:

- README.md -- Instrucciones de uso del programa.
- img/ -- Carpeta que contiene las imágenes utilizadas para el entrenamiento del modelo y el csv.
- requirements.txt -- Fichero con los paquetes necesarios para lanzar el programa.
- retina.py -- Código principal que ejecuta programa.
- train.py -- Código donde se realiza el entrenamiento del modelo.


## Instrucciones de uso:

1. Instalar requirements.txt.
```
pip install -r requirements.txt
```
2. Lanzar retina.py, a continuación elegir una imagen y darle aceptar
```
python3 retina.py
```
(en caso que se quiera reentrenar el modelo, habría que lanzar train.py y crear el directorio de imágenes con las imágenes de training que se encuentran en https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification/data )
