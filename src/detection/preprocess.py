import cv2
import numpy as np


class Preprocess():
    """
    Preprocesa la imagen para poder aplicar algoritmos sobre ella.
    Se recortan los bordes y se devuelve la imagen recortada
    y una máscara de ella con los bordes dilatados.
    """
    def __init__(self):
        pass

    def removeBlackBackground(self, gray_crop):
        """
        Eliminación de las franjas superiror e inferior de la imagen.

        :param gray_crop: imagen cortada
        :return: imagen cortada
        """
        thresholdBlack = 90
        
        meanColor = np.mean(gray_crop)
        for i in range(gray_crop.shape[1]):
            for j in range(gray_crop.shape[0]):
                if(gray_crop[j][i] < thresholdBlack):
                    gray_crop[j][i] = meanColor
                else:
                    gray_crop[j][i] = meanColor
                    break

        for i in range(gray_crop.shape[1]):
            for j in range(gray_crop.shape[0]):
                row_index = gray_crop.shape[0] - j - 1
                col_index = gray_crop.shape[1] - i - 1
                if(gray_crop[row_index][col_index] < thresholdBlack):
                    gray_crop[row_index][col_index] = meanColor
                else:
                    gray_crop[row_index][col_index] = meanColor
                    break

        return gray_crop

    def canny_filter(self, img):
        """
        Filtrado de canny para búsqueda de bordes

        :param img: imagen
        :return: imagen de bordes
        """
        mean = np.mean(img)
        img[img > mean*0.5] = 0
        return cv2.Canny(img, 50, 100)

    def crop_image(self, img, contours):
        """
        Recorte de la imagen

        :param img: imagen
        :param contours: contornos
        :return: imagen cortada
        """
        
        percentageToCrop = 0.15
        
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        yCrop = y+int(h*percentageToCrop)
        xCrop = x+int(w*percentageToCrop)
        return img[
                    y+int(h*percentageToCrop):y+h-int(h*percentageToCrop),
                    x+int(w*percentageToCrop):x+w-int(w*percentageToCrop)
                    ],xCrop, yCrop

    def get_contours(self, img):
        """
        Obtención de contornos

        :param img: imagen
        :return: lista de contornos
        """
        _, thresh = cv2.threshold(img, 100, 150, cv2.THRESH_BINARY)
        return cv2.findContours(
                                thresh,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    def convert_image_to_gray(self, img):
        """
        Método para convertir la imagen a escala de grises

        :param img: imagen de entrada en BGR
        :param th: threshold
        :return: imagen en escala de grises
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img_gray

    def morphology(self, type_m, img_gray, kernel_size=3, iterations=1):
        """
        Operación morfológica de erosión

        :param img_gray: imagen en escala de grises
        :param kernel_size: tamaño del elemento estructurante
        :return: imagen erosionada/dilatada
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if type_m == 'erode':
            return cv2.erode(img_gray, kernel, iterations=iterations)
        elif type_m == 'dilate':
            return cv2.dilate(img_gray, kernel, iterations=iterations)
