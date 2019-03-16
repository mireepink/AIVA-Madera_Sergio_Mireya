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

    def _applyHough(self, img_crop, img_dilation, path_im):
        """
        Aplica Hough sobre una imagen dada

        :param img_crop: imagen recortada
        :param img_dilation: máscara con bordes dilatados
        :param path_im: path de entrada de la imagen
        :return:
        """
        img_crop_return = img_crop.copy()
        minLineLength = 200
        maxLineGap = 100
        lines = cv2.HoughLinesP(img_dilation, 1, np.pi/180, 100, minLineLength,
                                maxLineGap)
        if (lines is not None):
            for line in lines:
                for x1, y1, x2, y2 in line:         
                    cv2.line(img_crop_return, (x1, y1), (x2, y2), (0, 0, 255), 8)
                      
                    
            return img_crop_return

    def selectRoi(self, colorImage, initImage, cropX, cropY):
        """
        Marca las regiones de interes sobre la imagen original de la madera
        
        :param colorImage: Imagen a color recortada con las grietas marcadas
        :param initImage: Imagen inicial de la madera
        :param cropX: Numero de pixeles que se han quitado de ancho de la
                      imagen origina al realizar el crop
        :param cropX: Numero de pixeles que se han quitado de alto de la
                      imagen origina al realizar el crop  
        :return initImage: Imagen inicial con las grietas marcadas
        """
        #print(colorImage)
        if type(colorImage) is np.ndarray: 
            img_hsv = cv2.cvtColor(colorImage, cv2.COLOR_BGR2HSV)
            lower_red = np.array([0,0,0])
            upper_red = np.array([10,255,255])
    
            mask = cv2.inRange(img_hsv, lower_red, upper_red)
            res = cv2.bitwise_and(colorImage,colorImage, mask= mask)
 
            res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            
            (thresh, im_bw) = cv2.threshold(res, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            kernel = np.ones((3,3), np.uint8)
            res = cv2.erode(res, kernel, iterations=1)
            res = cv2.dilate(res, kernel, iterations=3)
            
            
            gauss = cv2.GaussianBlur(res, (5,5), 0)
            
            t, dst = cv2.threshold(gauss, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE) 
            _, contours, _ = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in contours:
                area = cv2.contourArea(c)
                if area > 1000 and area < 10000:
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(initImage, (cropX + x, cropY + y), (cropX + x + w,cropY + y + h), (0, 0, 255), 4, cv2.LINE_AA)

            return initImage
        
            
    def saveImage(self, image, path_im):
        """
        Guarda la imagen resultado en el directorio indicado
        
        :param image: Imagen resultado que hay que guardar
        :path_im: Directorio donde guardar la imagen
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
        thresholdGray = 200
        
        img = self.load_im(path_im)
        img_gray = self.preprocess.convert_image_to_gray(img)
        img_gray[img_gray > thresholdGray] = 0
        image_erode = self.preprocess.morphology('erode', img_gray)
        contours = self.preprocess.get_contours(image_erode)
        img_crop, xCrop, yCrop = self.preprocess.crop_image(img, contours)
        gray_crop = self.preprocess.convert_image_to_gray(img_crop)
        gray_crop = self.preprocess.removeBlackBackground(gray_crop)
        edges = self.preprocess.canny_filter(gray_crop)
        img_dilated = self.preprocess.morphology(
                                                'dilate',
                                                edges,
                                                kernel_size=2,
                                                iterations=3)
        img_crop_line = self._applyHough(img_crop, img_dilated, path_im)
        imageWithROI = self.selectRoi(img_crop_line, img, xCrop, yCrop)
        self.saveImage(imageWithROI, path_im)
         
         
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Main parser')
    ap.add_argument('--path_im',
                    default='D:/Google Drive/MOVA/2_Cuatri/Aplicaciones/Trabajo/wood/original')
    ap.add_argument('--path_out', default='./out')
    FLAGS = ap.parse_args()
    if not os.path.exists(FLAGS.path_out):
        os.makedirs(FLAGS.path_out)

    for filename in glob.glob(FLAGS.path_im, '*'):
        deteccion = Deteccion(FLAGS.path_out)
        deteccion.process(filename)



