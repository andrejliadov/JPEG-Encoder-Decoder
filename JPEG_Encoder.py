import numpy as np
from PIL import Image
import matplotlib.image as img
import matplotlib.pyplot as plt

class Encoder:
    
    # Makes an NxN DCT matrix
    def constructDCT(self, size):
        self.size = size
        self.DCT = np.zeros(shape = (self.size, self.size))

        for i in range(0, self.size):
            if i == 0:
                for j in range(0, self.size):
                    self.DCT[i][j] = (1/np.sqrt(self.size))
            
            else:
                for j in range(0, self.size):
                    self.DCT[i][j] = np.sqrt(2.0/self.size) * np.cos( ((2*j+1.0)*i*np.pi) / (2.0 * self.size) )

                
    def printDCT(self):
        print(self.DCT)
    
    # Reads in an image and stores it as a numpy matrix
    def readFile(self, fileName):
        self.fileName = fileName
        self.image = img.imread(fileName)
        imgplot = plt.imshow(self.image)

    # This function needs to convert the RGB value into YUV 
    #def convertIamgeYUV():
        

    # This function encodes every NxN block of the image
    #def encodeImage():
        


encoder = Encoder()
encoder.constructDCT(8)
encoder.printDCT()
encoder.readFile('lena.png')
