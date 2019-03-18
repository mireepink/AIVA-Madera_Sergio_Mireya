import cv2
import numpy as np


class Preprocess():
    """
    Preprocesa la imagen para poder aplicar algoritmos sobre ella.
    Se recortan los bordes y se devuelve la imagen recortada
    y una mascara de ella con los bordes dilatados.
    """

    def __init__(self):
        pass

    def remove_black_background(self, gray_crop, threshold_black=90):
        """
        Eliminacion de las franjas superiror e inferior de la imagen.

        :param gray_crop: imagen cortada
        :param threshold_black: Umbral para el valor de 'negro' en los pixeles
        :return: imagen cortada
        """
        mean_color = np.mean(gray_crop)
        for i in range(gray_crop.shape[1]):
            for j in range(gray_crop.shape[0]):
                gray_crop[j][i] = mean_color
                if gray_crop[j][i] > threshold_black:
                    break

        for i in range(gray_crop.shape[1]):
            for j in range(gray_crop.shape[0]):
                row_index = gray_crop.shape[0] - j - 1
                col_index = gray_crop.shape[1] - i - 1
                gray_crop[row_index][col_index] = mean_color
                if gray_crop[row_index][col_index] > threshold_black:
                    break

        return gray_crop

    def canny_filter(self, img, th_mean=0.5, th_1=50, th_2=100):
        """
        Filtrado de canny para busqueda de bordes

        :param img: imagen
        :param th_mean: umbral para el valor de la media de los pixeles
        :param th_1: umbral inferior de Canny
        :param th_2: umbral superior de Canny
        :return: imagen de bordes
        """
        mean = np.mean(img)
        img[img > mean * th_mean] = 0
        return cv2.Canny(img, th_1, th_2)

    def crop_image(self, img, rect, percentage_to_crop = 0.15):
        """
        Recorte de la imagen en funcion de los valors
        obtenidos anteriormente en la funcion get_contours

        :param img: imagen
        :param rect: coordenadas del rectangulo a recortar
        :param percentage_to_crop porcentaje extra a cortar
        :return: imagen cortada
        """
        x, y, w, h = rect
        y_crop = y + int(h * percentage_to_crop)
        x_crop = x + int(w * percentage_to_crop)
        return img[
               y + int(h * percentage_to_crop):y + h - int(h * percentage_to_crop),
               x + int(w * percentage_to_crop):x + w - int(w * percentage_to_crop)
               ], x_crop, y_crop

    def get_contours(self, img, threshold=100, max_value=150):
        """
        Obtencion de contornos

        :param img: imagen
        :param threshold: valor del umbral
        :param max_value: valor maximo del umbral
        :return: lista de contornos
        """
        _, thresh = cv2.threshold(img, threshold, max_value, cv2.THRESH_BINARY)

        contours = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

        return cv2.boundingRect(contours[0])

    def convert_image_to_gray(self, img):
        """
        Metodo para convertir la imagen a escala de grises

        :param img: imagen de entrada en BGR
        :param th: threshold
        :return: imagen en escala de grises
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img_gray

    def morphology(self, type_m, img_gray, kernel_size=3, iterations=1):
        """
        Operacion morfologica de erosion

        :param type_m: Tipo de operacion a realizar
        :param img_gray: imagen en escala de grises
        :param kernel_size: tamano del elemento estructurante
        :param iterations: Numero de iteracciones a realizar
        :return: imagen erosionada/dilatada
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if type_m == 'erode':
            return cv2.erode(img_gray, kernel, iterations=iterations)
        elif type_m == 'dilate':
            return cv2.dilate(img_gray, kernel, iterations=iterations)
