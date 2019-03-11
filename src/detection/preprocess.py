import cv2
import numpy as np


class Preprocess():
    """
    Preprocesa la imagen para poder aplicar algoritmos sobre ella.
    Preprocesado de la imagen donde se recortan los bordes y se devuelve
    la imagen recortada y una máscara de ella con los bordes dilatados.

    """
    def __init__(self):
        pass

    def removeBlackBackground(self, gray_crop):
        """
        Eliminar las franjas superiror e inferior de la imagen.

        :param gray_crop:
        :return:
        """
        for i in range(gray_crop.shape[1]):
            for j in range(gray_crop.shape[0]):
                if (gray_crop[j][i] < 90):
                    gray_crop[j][i] = 183
                else:
                    gray_crop[j][i] = 183
                    break

        for i in range(gray_crop.shape[1]):
            for j in range(gray_crop.shape[0]):
                if (gray_crop[gray_crop.shape[0] - j - 1][gray_crop.shape[1] - i - 1] < 90):
                    gray_crop[gray_crop.shape[0] - j - 1][gray_crop.shape[1] - i - 1] = 183
                else:
                    gray_crop[gray_crop.shape[0] - j - 1][gray_crop.shape[1] - i - 1] = 183
                    break

        return gray_crop

    def canny_filter(self, img):
        """

        :param img:
        :return:
        """
        mean = np.mean(img)
        img[img > mean*0.5] = 0
        return cv2.Canny(img, 50, 100)

    def crop_image(self, img, contours):
        """

        :param img:
        :param contours:
        :return:
        """
        cnt = contours[0][0]
        x, y, w, h = cv2.boundingRect(cnt)
        return img[
                    y+int(h*0.15):y+h-int(h*0.15),
                    x+int(w*0.15):x+w-int(w*0.15)
                    ]

    def get_contours(self, img):
        """

        :param img:
        :return:
        """
        _, thresh = cv2.threshold(img, 100, 150, cv2.THRESH_BINARY)
        return cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def convert_image_to_gray(self, img):
        """
        Método para convertir la imagen a escala de grises

        :param img: imagen de entrada en BGR
        :type img: nd.array
        :param th: threshold
        :type th: int
        :return: imagen en escala de grises
        :rtype: nd.array
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img_gray


    def morphology(self, type_m, img_gray, kernel_size = 3, iterations = 1):
        """
        Operación morfológica de erosión

        :param img_gray: imagen en escala de grises
        :type img_gray: nd.array (:,:,1)
        :param kernel_size: tamaño del elemento estructurante
        :type kernel_size: int
        :return: eroded image
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if type_m == 'erode':
            return cv2.erode(img_gray, kernel, iterations=iterations)
        elif type_m == 'dilate':
            return cv2.dilate(img_gray, kernel, iterations=iterations)
