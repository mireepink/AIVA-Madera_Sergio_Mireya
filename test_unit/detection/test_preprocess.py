import os
import cv2

from unittest import TestCase
from src.detection.preprocess import Preprocess

class Test_Preprocess(TestCase):
    def setUp(self):
        self.thresholdGray = 200
        self.path_im = os.path.join(os.pardir, 'data/st1456')
        self.path_im_gray = os.path.join(os.pardir, 'data/st1456_gray.png')
        self.path_im_gray_erode = os.path.join(os.pardir, 'data/st1456_gray_erode.png')
        self.path_im_gray_dilate = os.path.join(os.pardir, 'data/st1456_gray_dilate.png')
        self.preprocess = Preprocess()

    def test_convert_image_to_gray(self):
        img = cv2.imread(self.path_im)
        img_gray_expected = cv2.imread(self.path_im_gray, 0)
        img_gray = self.preprocess.convert_image_to_gray(img)
        self.assertTrue((img_gray_expected == img_gray).all(), True)

    def test_morpholofy(self):
        img_gray = cv2.imread(self.path_im_gray, 0)
        img_gray[img_gray > self.thresholdGray] = 0
        img_gray_erode_expected = cv2.imread(self.path_im_gray_erode, 0)
        image_erode = self.preprocess.morphology('erode', img_gray)
        self.assertTrue((img_gray_erode_expected == image_erode).all(), True)
        img_gray_dilate_expected = cv2.imread(self.path_im_gray_dilate, 0)
        image_dilate = self.preprocess.morphology('dilate', img_gray)
        self.assertTrue((img_gray_dilate_expected == image_dilate).all(), True)


