import os
import cv2
import numpy as np

from unittest import TestCase
from src.detection.preprocess import Preprocess

class TestPreprocess(TestCase):
    def setUp(self):
        self.thresholdGray = 200

        self.path_im = os.path.join(os.pardir, 'AIVA-Madera_Sergio_Mireya/test_unit/data/st1456')
        self.path_im_gray = os.path.join(os.pardir, 'AIVA-Madera_Sergio_Mireya/test_unit/data/st1456_gray.png')
        self.path_im_gray_erode = os.path.join(os.pardir, 'AIVA-Madera_Sergio_Mireya/test_unit/data/st1456_gray_erode.png')
        self.path_im_gray_dilate = os.path.join(os.pardir, 'AIVA-Madera_Sergio_Mireya/test_unit/data/st1456_gray_dilate.png')
        self.rect = (0, 116, 488, 396)
        self.shape_image_cropped = (278, 342, 3)
        self.n_edges = 1042
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

    def test_get_contours(self):
        img_gray = cv2.imread(self.path_im_gray_erode, 0)
        x, y, w, h = self.preprocess.get_contours(img_gray, threshold = 100, max_value = 150)
        self.assertEqual((x, y, w, h), self.rect)

    def test_crop_image(self):
        img = cv2.imread(self.path_im)
        img_crop, x_crop, y_crop = self.preprocess.crop_image(img, self.rect, percentage_to_crop=0.15)
        self.assertEqual(img_crop.shape, self.shape_image_cropped)
        self.assertEqual(x_crop, 73)
        self.assertEqual(y_crop, 175)

    def test_canny_filter(self):
        img_gray_erode = cv2.imread(self.path_im_gray_erode, 0)
        edges = self.preprocess.canny_filter(img_gray_erode, th_mean = 0.5, th_1 = 50, th_2 = 100)
        self.assertEqual(edges[edges == 255].size, self.n_edges)
        edges = self.preprocess.canny_filter(img_gray_erode, th_mean = 0.4, th_1 = 55, th_2 = 102)
        self.assertNotEqual(edges[edges == 255].size, self.n_edges)

    def test_remove_black_background(self):
        img_gray = cv2.imread(self.path_im_gray_erode, 0)
        img_no_background = self.preprocess.remove_black_background(img_gray, threshold_black=90)
        self.assertEqual(type(img_no_background), np.ndarray)
        self.assertEqual(np.mean(img_no_background), 61.0)




