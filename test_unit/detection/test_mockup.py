import os
import numpy as np

from unittest import TestCase
from src.detection.mockup import Deteccion_Mockup

class Test_Deteccion_Mockup(TestCase):
    def setUp(self):
        self.deteccion = Deteccion_Mockup()
        self.path_im = os.path.join(os.pardir, 'data/st1456')
        self.sum_im_expected = 104339933
        self.im_shape_expected = (512, 488, 3)

    def aux_assert_im(self, im):
        sum_im_result = np.sum(im)
        im_shape_result = im.shape
        self.assertEqual(self.sum_im_expected, sum_im_result)
        self.assertEqual(self.im_shape_expected, im_shape_result)

    def test_load_im(self):
        im = self.deteccion.load_im(self.path_im)
        self.aux_assert_im(im)

    def test_detecta(self):
        im, bbox = self.deteccion.detecta(self.path_im)
        self.aux_assert_im(im)
        self.assertEqual(len(bbox), 4)

    def test_rect(self):
        self.deteccion.dx = -30
        self.deteccion.dy = 10
        im, bbox = self.deteccion.detecta(self.path_im)
        self.assertIsNone(bbox)
