import cv2
import argparse
import numpy as np
import glob
import os

from preprocess import Preprocess
# from src.detection.preprocess import Preprocess


class Deteccion():
    def __init__(self, path_out):
        self.path_out = path_out
        self.preprocess = Preprocess()

    def load_im(self, path_im):
        """
        Lee la imagen de entrada

        :param path_im: path de la imagen de entrada
        :return: imagen
        """
        im = cv2.imread(path_im)
        return im

    def applyHough(self, img_crop, img_dilation, path_im):
        """
        Aplica Hough sobre una imagen dada

        :param img_crop: imagen recortada
        :param img_dilation: máscara con bordes dilatados
        :param path_im: path de entrada de la imagen
        :return:
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
            cv2.imwrite(
                        os.path.join(self.path_out, file + '_split.jpg'),
                        img_crop)

    def process(self, path_im):
        """
        Proceso principal

        :param path_im: path de la imagen a procesar
        """
        img = self.load_im(path_im)
        img_gray = self.preprocess.convert_image_to_gray(img)
        img_gray[img_gray > 200] = 0
        image_erode = self.preprocess.morphology('erode', img_gray)
        contours = self.preprocess.get_contours(image_erode)
        img_crop = self.preprocess.crop_image(img, contours)
        gray_crop = self.preprocess.convert_image_to_gray(img_crop)
        gray_crop = self.preprocess.removeBlackBackground(gray_crop)
        edges = self.preprocess.canny_filter(gray_crop)
        img_dilated = self.preprocess.morphology(
                                                'dilate',
                                                edges,
                                                kernel_size=2,
                                                iterations=3)
        self.applyHough(img_crop, img_dilated, path_im)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Main parser')
    ap.add_argument('--path_im',
                    default='D:/Google Drive/MOVA/2_Cuatri/Aplicaciones/Trabajo/wood/original')
    ap.add_argument('--path_out', default='./out')
    FLAGS = ap.parse_args()

    if not os.path.exists(FLAGS.path_out):
        os.makedirs(FLAGS.path_out)

    for filename in glob.glob(os.path.join(FLAGS.path_im, '*')):
        deteccion = Deteccion(FLAGS.path_out)
        deteccion.process(filename)
