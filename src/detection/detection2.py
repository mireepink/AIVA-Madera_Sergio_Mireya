import cv2
import argparse
import numpy as np
import glob
import os


class Deteccion():
    def __init__(self):
        self.x = 387
        self.dx = 30
        self.y = 110
        self.dy = 150

    def load_im(self, path_im):
        """Lee la imagen de entrada

        Parámetros de entrada:
        path_im -- Path de la imagen de entrada
        """
        im = cv2.imread(path_im)
        return im

    def show(self, im, bbox):
        """Muestra la ROI de donde se encuentra la grieta

        Parámetros de entrada:
        im -- Imagen de entrada
        bbox -- Coordenadas de la ROI
        """
        cv2.putText(im, "HAY GRIETA", (350, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(
                        im, (bbox[0], bbox[2]),
                        (bbox[0] + bbox[1], bbox[2] + bbox[3]),
                        (0, 255, 0), 2
                        )
        cv2.imshow('Deteccion', im)
        cv2.waitKey()

    def preprocess(self, path_im):
        """ Preprocesa la imagen para poder aplicar algoritmos sobre ella.

        Preprocesado de la imagen donde se recortan los bordes y se devuelve
        la imagen recortada y una máscara de ella con los bordes dilatados.

        Parámetros de entrada:
        path_im -- Path de la imagen de entrada

        Parámetros de salida:
        img_crop -- Imagen recortada
        img_dilation -- Máscara con bordes dilatados
        """
        img = self.load_im(path_im)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray[gray > 200] = 0

        kernel_erode = np.ones((3, 3), np.uint8)
        img_erode = cv2.erode(gray, kernel_erode, iterations=1)

        _, thresh = cv2.threshold(img_erode, 100, 150, cv2.THRESH_BINARY)

        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        img_crop = img[
                        y+int(h*0.15):y+h-int(h*0.15),
                        x+int(w*0.15):x+w-int(w*0.15)
                        ]

        gray_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray_crop)
        gray_crop[gray_crop > mean*0.5] = 0

        edges = cv2.Canny(gray_crop, 50, 100)
        kernel = np.ones((2, 2), np.uint8)

        img_dilation = cv2.dilate(edges, kernel, iterations=3)

        return img_crop, img_dilation

    def applyHough(self, img_crop, img_dilation, path_im):
        """ Aplica Hough sobre una imagen dada.

        Parámetros de entrada:
        img_crop -- Imagen recortada
        img_dilation -- Máscara con bordes dilatados
        path_im -- Path de entrada de la imagen
        """

        minLineLength = 200
        maxLineGap = 100
        lines = cv2.HoughLinesP(img_dilation, 1, np.pi/180, 100, minLineLength,
                                maxLineGap)
        if (lines is not None):
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(img_crop, (x1, y1), (x2, y2), (0, 0, 255), 4)
            file = path_im.split('/')[-1].split('\\')[-1]
            cv2.imwrite('out/' + file + '_split.jpg', img_crop)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Main parser')
    mkdir_folder = 'out'
    if not os.path.exists(mkdir_folder):
        os.makedirs(mkdir_folder)
    for filename in glob.glob(
                                'D:/Google Drive/MOVA/2_Cuatri/Aplicaciones/' +
                                'Trabajo/wood/original/*'):

        deteccion = Deteccion()

        img_crop, img_dilation = deteccion.preprocess(filename)
        deteccion.applyHough(img_crop, img_dilation, filename)
