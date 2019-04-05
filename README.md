# AIVA-Madera_Sergio_Mireya
* Jorge Vela Peña
* Sergio Domínguez Rodríguez
* Mireya Elisabet Valenzuela Carballo
 
## Descripción
Proyecto de detección de grietas en la madera. MOVA URJC.
Este proyecto inspeccionará la madera en búsqueda de grietas de forma autónoma. La realización de este proyecto supone numerosos beneficios y ventajas para la problemática industrial de detección de grietas en superficies. En nuestro caso, el enfoque se llevará a cabo en tablas de madera. 
El sistema propuesto supondrá la obtención de estructuras de palés o tablas más robustas y seguras, lo que incrementará su calidad final. 
Este tipo de estructuras más consistentes permitirá abrir el abanico a un mayor número de futuros clientes, al conseguir afrontar cargas más pesadas con una mayor estabilidad.
Además, se conseguirá ahorrar en mano de obra y en tiempo empleado en la detección, al tratarse de un sistema de visión autónomo.

El desarrollo de este algoritmo tiene el objetivo de detectar grietas en la madera y marcarlas sobre la imagen.

Para poder ejecutarlo, son necesarios los siguientes pasos:

## Para ejecutarlo en LOCAL:
### Preparar entorno
```
git clone https://github.com/mireepink/AIVA-Madera_Sergio_Mireya.git
cd AIVA-Madera_Sergio_Mireya
export PYTHONPATH=$PYTHONPATH:'pwd'
```

### Instalar requerimientos
* Python 3.6.4
* opencv-python==3.4.5.20
* numpy==1.16.1
* pytest==4.2.0
* pytest-cov==2.6.1
```
pip install -r requirements.txt
```

### Lanzar programa
```
python src/detection/detection.py --path_im [directorio con las imágenes de entrada] --path_out [directorio para las imágenes de salida]
```
Ejemplo: python src/detection/detection.py --path_im /Users/mireepinki/Downloads/wood/original --path_out ./out

### Lanzar tests unitarios
```
pytest --cov=src test_unit/*/test_*
```
## Para ejecutarlo mediante DOCKER:
### Descargar nuestra imagen docker
```
docker pull smjmuva/aiva_wood_group_2:latest
```
### Ejecutar el docker
```
docker run --rm -v [directorio con las imágenes de entrada]:/INPUTS -v [directorio para las imágenes de salida]:/OUTPUTS smjmuva/aiva_wood_group_2:latest
```
Ejemplo: docker run --rm -v /home/sergio/Descargas/wood/original:/INPUTS -v /home/sergio/Descargas/out_wood:/OUTPUTS smjmuva/aiva_wood_group_2:latest

### Pasos para generar localmente la imagen docker:
Si quieres generarte la imagen docker a partir del código ya descargado puedes hacerlo con los siguientes pasos:
Te creas un fichero que se llame Dockerfile (sin extensión), este fichero debe estar justo un nivel más arriba que la carpeta con el código del proyecto, y pegas en él las siguientes líneas:
````
FROM python:3
ADD ./AIVA-Madera_Sergio_Mireya /AIVA-Madera_Sergio_Mireya
WORKDIR /AIVA-Madera_Sergio_Mireya
RUN pip install -r requirements.txt
ENV PYTHONPATH $PYTHONPATH: 'pwd'
ENTRYPOINT ["python", "src/detection/detection.py"]
````
Con estas líneas lo que estamos diciendo es que queremos crear una imagen docker que parta de una imagen docker de Python, en concreto la versión 3 de Python. Posteriormente añadimos la carpeta del código del proyecto a la imagen docker. Cambiamos el directorio de trabajo a la ruta del proyecto. Instalamos todas las dependencias de las librerías que tiene nuestro proyecto. Configuramos la variable de entorno de Python. Y por último, le decimos que queremos ejecutar dicho fichero Python.

Una vez tenemos creado nuestro fichero Dockerfile, ejecutamos en ese directorio el siguiente comando:
```
sudo docker build -t [nombre de la imagen] .
```
Ejemplo: sudo docker build -t wood_docker .

Con esto generamos la imagen docker con los requisitos que se han definido en el Dockerfile.

Una vez generada la imagen docker se puede ejecutar tal y como hemos descrito anteriormente.
