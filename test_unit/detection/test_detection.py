import os
import cv2
import numpy as np

from unittest import TestCase
from src.detection.detection import Detection


class TestDetection(TestCase):
    def set_up(self):
        self.path_out = os.path.join(os.pardir, 'AIVA-Madera_Sergio_Mireya/test_unit/data')
        self.path_im1 = os.path.join(os.pardir, 'AIVA-Madera_Sergio_Mireya/test_unit/data/st1526')
        self.path_im2 = os.path.join(os.pardir, 'AIVA-Madera_Sergio_Mireya/test_unit/data/st1456')
        self.path_im1_out = os.path.join(os.pardir, 'AIVA-Madera_Sergio_Mireya/test_unit/data/st1526_split.jpg')
        self.path_im2_out = os.path.join(os.pardir, 'AIVA-Madera_Sergio_Mireya/test_unit/data/st1456_split.jpg')
        self.shape_out_expected = (512, 488, 3)
        self.detection = Detection(self.path_out)

    def test_process(self):
        self.detection.process(self.path_im1)
        self.assertTrue(os.path.isfile(self.path_im1_out))
        self.detection.process(self.path_im2)
        self.assertFalse(os.path.isfile(self.path_im2_out))
        shape_out = cv2.imread(self.path_im1_out).shape
        self.assertEqual(shape_out, self.shape_out_expected)
