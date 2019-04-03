import cv2
import argparse
import numpy as np
import glob
import os

from src.detection.preprocess import Preprocess


class Detection:
    def __init__(self, path_out):
        self.path_out = path_out
        self.preprocess = Preprocess()

    def _load_im(self, path_im):
        """
        Lee la imagen de entrada

        :param path_im: path de la imagen de entrada
        :return: imagen
        """
        im = cv2.imread(path_im)
        return im

    def _apply_hough(self, img_crop, img_dilation):
        """
        Aplica Hough sobre una imagen dada

        :param img_crop: imagen recortada
        :param img_dilation: mascara con bordes dilatados
        :param path_im: path de entrada de la imagen
        :return:
        """
        img_crop_return = img_crop.copy()
        threshold = 100
        lines = 1
        min_line_length = 200
        max_line_gap = 100

        color_line_b = 0
        color_line_g = 0
        color_line_r = 255
        size_line = 8
        lines = cv2.HoughLinesP(img_dilation, lines, np.pi/180, threshold, min_line_length,
                                max_line_gap)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(img_crop_return, (x1, y1), (x2, y2), (color_line_b, color_line_g, color_line_r), size_line)

            return img_crop_return

    def _select_roi(self, color_image, init_image, crop_x, crop_y):
        """
        Marca las regiones de interes sobre la imagen original de la madera

        :param color_image: Imagen a color recortada con las grietas marcadas
        :param init_image: Imagen inicial de la madera
        :param crop_x: Numero de pixeles que se han quitado de ancho de la
                      imagen origina al realizar el crop
        :param crop_y: Numero de pixeles que se han quitado de alto de la
                      imagen origina al realizar el crop
        :return initImage: Imagen inicial con las grietas marcadas
        """
        if type(color_image) is np.ndarray:
            img_hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            lower_red = np.array([0, 0, 0])
            upper_red = np.array([10, 255, 255])

            mask = cv2.inRange(img_hsv, lower_red, upper_red)
            res = cv2.bitwise_and(color_image, color_image, mask=mask)
            res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            threshold = 150
            max_val = 255
            (thresh, im_bw) = cv2.threshold(res, threshold, max_val, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            kernel_erode_size = 3
            kernel = np.ones((kernel_erode_size, kernel_erode_size), np.uint8)
            res = cv2.erode(res, kernel, iterations=1)
            res = cv2.dilate(res, kernel, iterations=3)

            kernel_gauss_size = 5
            gauss = cv2.GaussianBlur(res, (kernel_gauss_size, kernel_gauss_size), 0)

            threshold = 0
            t, dst = cv2.threshold(gauss, threshold, max_val, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
            _, contours, _ = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            color_line_b = 0
            color_line_g = 0
            color_line_r = 255
            line_type = 4
            for c in contours:
                area = cv2.contourArea(c)
                min_area = 1000
                max_area = 10000
                if area > min_area and area < max_area:
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(
                                    init_image,
                                    (crop_x + x, crop_y + y),
                                    (crop_x + x + w, crop_y + y + h),
                                    (color_line_b, color_line_g, color_line_r),
                                    line_type,
                                    cv2.LINE_AA
                                    )

            return init_image

    def _save_image(self, image, path_im):
        """
        Guarda la imagen resultado en el directorio indicado

        :param image: Imagen resultado que hay que guardar
        :param path_im: Directorio donde guardar la imagen
        """
        if type(image) is np.ndarray:
            file = path_im.split('/')[-1].split('\\')[-1]
            cv2.imwrite(
                        os.path.join(self.path_out, file + '_split.jpg'),
                        image)

    def process(self, path_im):
        """
        Proceso principal

        :param path_im: path de la imagen a procesar
        """

        img = self._load_im(path_im)
        crop_data, img_dilated = self.preprocess.preprocess_image(img)

        img_crop = crop_data[0]
        x_crop = crop_data[1]
        y_crop = crop_data[2]

        img_crop_line = self._apply_hough(img_crop, img_dilated)
        image_with_roi = self._select_roi(img_crop_line, img, x_crop, y_crop)
        self._save_image(image_with_roi, path_im)


if __name__ == '__main__':  # pragma: no cover
    ap = argparse.ArgumentParser(description='Main parser')
    ap.add_argument('--path_im', default='/INPUTS')
    ap.add_argument('--path_out', default='/OUTPUTS')
    FLAGS = ap.parse_args()

    if not os.path.exists(FLAGS.path_out):
        os.makedirs(FLAGS.path_out)

    total_img = str(len(glob.glob(os.path.join(FLAGS.path_im, '*'))))
    print("Encontradas " + total_img + " en el directorio")
    img_count = 1

    for filename in glob.glob(os.path.join(FLAGS.path_im, '*')):
        detection = Detection(FLAGS.path_out)
        detection.process(filename)

    total_img_split = str(len(glob.glob(os.path.join(FLAGS.path_out, '*'))))
    print("Encontradas " + total_img_split + " imagenes con grietas")