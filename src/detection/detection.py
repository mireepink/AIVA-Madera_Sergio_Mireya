import cv2
import argparse

class Deteccion():
    def __init__(self):
        self.x = 387
        self.dx = 30
        self.y = 110
        self.dy = 150

    def load_im(self, path_im):
        im = cv2.imread(path_im)
        return im

    def show(self, im, bbox):
        cv2.putText(im, "HAY GRIETA", (350,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,255),2,cv2.LINE_AA)
        cv2.rectangle(im,(bbox[0],bbox[2]),(bbox[0] + bbox[1],bbox[2] + bbox[3]), (0, 255, 0), 2)
        cv2.imshow('Deteccion', im)
        cv2.waitKey()

    def detecta(self, path_im):
        im = self.load_im(path_im)
        bbox = [self.x, self.dx, self.y, self.dy]

        if self.dx < 0 or self.dy < 0:
            return im, None

        return im, bbox


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Main parser')
    ap.add_argument('--path_im', default='/Users/mireepinki/Downloads/wood/original/st1456')
    FLAGS = ap.parse_args()

    deteccion = Deteccion()

    im, bbox = deteccion.detecta(FLAGS.path_im)
    deteccion.show(im, bbox)

