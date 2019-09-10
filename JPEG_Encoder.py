import numpy as np
from PIL import Image
import matplotlib.image as img
import matplotlib.pyplot as plt

class Encoder:
    
    Qlum = [[16, 11, 10, 16, 24, 40, 51, 61], 
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120,101],
            [72, 92, 95, 98, 112, 100, 103, 99]]
    
    Qchr = [[17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 16, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]]
    
    # Makes an NxN DCT matrix
    def constructDCT(self, size):
        self.size = size
        self.DCT = np.zeros(shape = (self.size, self.size))

        for i in range(0, self.size):
            if i == 0:
                for j in range(0, self.size):
                    self.DCT[i][j] = (1.0/np.sqrt(self.size))
            
            else:
                for j in range(0, self.size):
                    self.DCT[i][j] = np.sqrt(2.0/self.size) * np.cos( ((2.0*j+1.0)*i*np.pi) / (2.0 * self.size) )

                
    def printDCT(self):
        print(self.DCT)
    
    # Reads in an image and stores it as a numpy matrix
    def readFile(self, fileName):
        self.fileName = fileName
        self.image = img.imread(fileName)

    def convertIamgeYUV():
        self.imageYUV = self.image.convert('YCbCr')
        
    # This functions does a piece wise matrix mult of the DCT and the image
    def applyDCT(self):
        self.imageEnc = np.zeros(self.image.shape)
        temp = np.zeros(self.DCT.shape)
        blockSize = self.DCT.shape[0]
        imgH, imgW, numChan = self.image.shape

        for chan in range(0, numChan):
            for row in range(0, imgH, blockSize):
                for col in range(0, imgW, blockSize):
                    temp = np.dot(self.image[row:row+blockSize, col:col+blockSize, chan], np.transpose(self.DCT))
                    self.imageEnc[row:row+blockSize, col:col+blockSize, chan] = np.dot(self.DCT, temp)

    # This step applies quantisation matrix according to JPEG standard
    def quantisation(self):
        self.quantisedImage = np.zeros(self.imageEnc.shape)
        temp = np.zeros(self.DCT.shape)
        blockSize = self.DCT.shape[0]
        imgH, imgW, numChan = self.image.shape

        for chan in range(0, numChan):
            for row in range(0, imgH, blockSize):
                for col in range(0, imgW, blockSize):
                    temp = self.imageEnc[row:row+blockSize, col:col+blockSize, chan]
                    self.quantisedImage[row:row+blockSize, col:col+blockSize, chan] = np.round((temp * 256) / self.Qlum)


    # This function has to seperate the transformed image into blocks and zig zag
    # Through it for the sake of encoding
    def zigZag(self):
        block = np.zeros(self.DCT.shape)
        blk = self.DCT.shape[0]
        imgH, imgW, numChan = self.quantisedImage.shape
        self.ACCoef = np.zeros(shape = ( (blk * blk) -1, 3))
        counter = 0

        for chan in range(0, numChan):
            for row in range(0, imgH, blk):
                for col in range(0, imgW, blk):
                    block = self.quantisedImage[row:row+blk, col:col+blk, chan]
                    i = 0
                    j = 1 
                    while j < blk and i < blk:
                        while j > 0:
                            self.ACCoef[counter, chan] = block[i, j]
                            counter += 1
                            print(counter)
                            j -= 1
                            i += 1
                            print(i, j)
                            
                        if i < blk - 1:
                            i += 1 
                            print(i, j)

                        if i == blk - 1:
                            j += 1
                            print(i, j)
                            self.ACCoef[counter, chan] = block[i, j]
                            counter += 1
                            print(counter)
                        
                        while i > 0 and j < blk:
                            self.ACCoef[counter, chan] = block[i, j]
                            counter += 1
                            print(counter)
                            i -= 1
                            j += 1
                            print(i, j)
                        
                        j += 1
                        print(i, j, 'here')

                        if j == blk - 1:
                            i += 1
                            print(i, j)
                            self.ACCoef[counter, chan] = block[i, j]
                            counter += 1
                            print(counter)

     

        
encoder = Encoder()
encoder.constructDCT(8)
encoder.printDCT()
encoder.readFile('lena.png')
encoder.applyDCT()
encoder.quantisation()
encoder.zigZag()
plt.show(encoder.imageEnc)
