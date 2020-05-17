import numpy as np
import random
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter

class CorpusCallosum:
    
    def __init__(self, img):
        self.img = img
        
    def Compute(self,img):
        noise_img = self.sp_noise(img,0.02)
        
        image = cv2.cvtColor(noise_img, cv2.COLOR_BGR2HSV) # convert to HSV
        figure_size = 9 # the dimension of the x and y axis of the kernal.
        new_image = cv2.blur(image,(figure_size, figure_size))
        new_image=cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)
        
        blurred = cv2.pyrMeanShiftFiltering(new_image,111,141)
        gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
        ret,threshold = cv2.threshold(gray,70,255,cv2.THRESH_BINARY)
        
        t_full = threshold
        rows = t_full.shape[0]
        columns = t_full.shape[1]
        for i in range(0,int(rows/2-33)):
            for j in range(0,columns):
                if(t_full[i,j] == 0):
                    t_full[i,j]=255
        for i in range(int(rows/2+50),rows):
             for j in range(0,columns):
                      if(t_full[i,j] == 0):
                              t_full[i,j]=255
        for i in range(0,rows):
            for j in range(0,int(columns/2-30)):
                if(t_full[i,j] == 0):
                       t_full[i,j]=255
        for i in range(0,rows):
            for j in range(int(columns/2+35),columns):
                if(t_full[i,j] == 0):
                        t_full[i,j]=255
        return t_full
    
    def sp_noise(self,image,prob):
        output = np.zeros(image.shape,np.uint8)
        thres = 1 - prob 
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output
    
    def plot_corpus_callosum(self):
        
        t_full = self.Compute(self.img)
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Extraction of Corpus callosum', fontsize=20)
    
        ax1.set_title('original')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.imshow(self.img)
    
        ax2.set_title('corpus callosum')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.imshow(t_full, cmap='gray')
    
        plt.subplots_adjust(top=.85)
        plt.show()
    
    def corpus_callosum_area(self):
        t_full = self.Compute(self.img)
        n = np.sum(t_full == 0)
        area = (n**0.5)*0.264
        return area

