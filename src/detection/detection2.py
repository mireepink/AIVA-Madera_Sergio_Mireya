import cv2
import argparse
import numpy as np
import glob
import os

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
        img = self.load_im(path_im)

        #Esta es la parte en la que se hace el crop de la imagen para quitar los negros (habria que meterlo en una funcion)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #Primero los pixeles que esten muy cerca de 0 los paso a negro, para quitarme las letras de arriba
        gray[gray>200] = 0
        
        #Como quedan aun algunas lineas suaves de los numeros de arriba le hago un erode, para cargarmelas
        kernel_erode = np.ones((3,3), np.uint8)
        img_erode = cv2.erode(gray, kernel_erode, iterations=1)
        
        #Con esta imagen que ya no le quedan letras arriba, hago un crop, con un rango de colores bastante altos para evitar el negro
        _,thresh = cv2.threshold(img_erode,100,150,cv2.THRESH_BINARY)

        contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)        
        img_crop = img[y:y+h,x:x+w]

        #Con la imagen en gris y recortada, calculo la media de los valores de los pixeles 
        # la multiplico por 0.5 y me quedo con los valores que son superiores
        gray_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray_crop)
        gray_crop[gray_crop > mean*0.5] = 0

        #Y con esto aplico Canny
        edges = cv2.Canny(gray_crop, 50, 100)
        kernel = np.ones((5,5), np.uint8)

        #A lo que me devuelve canny le hago un poco de dilatacion para agrandar las grietas
        # para que sean continuas y se unan los trocitos
        img_dilation = cv2.dilate(edges, kernel, iterations=10)

        #Y con esta imagen, aplico Hough
        minLineLength = 200
        maxLineGap = 100
        lines = cv2.HoughLinesP(img_dilation,1,np.pi/180,100,minLineLength,maxLineGap)
        if (lines is not None):
            for line in lines:
                for x1,y1,x2,y2 in line:
                    cv2.line(img_crop,(x1,y1),(x2,y2),(0,0,255),1)
            file =path_im.split('/')[-1].split('\\')[-1]
            cv2.imwrite('out/' + file + '_split.jpg',img_crop)
        

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Main parser')
    mkdir_folder = 'out'
    if not os.path.exists(mkdir_folder):
        os.makedirs(mkdir_folder)
    for filename in glob.glob('D:/Google Drive/MOVA/2_Cuatri/Aplicaciones/Trabajo/wood/original/*'):
 
        deteccion = Deteccion()

        deteccion.detecta(filename)

