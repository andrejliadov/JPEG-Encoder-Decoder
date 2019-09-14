import numpy as np
from PIL import Image
import matplotlib.image as img
import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy.testing

def reverseInt(num):
    return int(str(x)[::-1])

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
        self.image = 255 * self.image
        self.image = self.image.astype(int)
        plt.imshow(self.image, norm=col.Normalize(0, 255))
        plt.show()

    def convertImageYUV(self):
        imgH, imgW, numChan = self.image.shape

        for chan in range(0,numChan):
            for row in range(0, imgH):
                for col in range(0, imgW):
                    if chan == 0:
                        self.image[row, col, chan] = (0.3 * self.image[row, col, 0]) + (0.6 * self.image[row, col, 1]) + (0.1 * self.image[row, col, 2])

                    elif chan == 1:
                        self.image[row, col, chan] = 0.5*( (self.image[row, col ,2]) - (0.3 * self.image[row, col, 0]) + (0.6 * self.image[row, col, 1]) + (0.1 * self.image[row, col, 2]) )

                    elif chan == 2:
                        self.image[row, col, chan] = 0.625 * ( (self.image[row, col, 0]) - ((0.3 * self.image[row, col, 0]) + (0.6 * self.image[row, col, 1]) + (0.1 * self.image[row, col, 2])) )
        
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
        plt.imshow(self.imageEnc[:,:,1])
        plt.show()

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
        plt.imshow(self.quantisedImage)
        plt.show()
    
    def zag(self):
        block = np.zeros(self.DCT.shape)
        blk = self.DCT.shape[0]
        imgH, imgW, numChan = self.quantisedImage.shape
        self.ACCoef = np.zeros(shape = ((blk * blk) - 1, 3))
        indexList = np.zeros(2, dtype=int)

        for chan in range(0, numChan):
            for row in range(0, imgH, blk):
                for col in range(0, imgW, blk):
                    block = self.quantisedImage[row:row+blk, col:col+blk, chan]
                    indexList[0] = 0
                    indexList[1] = 1
                    rev = indexList[::-1]
                    counter = 0
                    
                    while indexList[0] < blk and indexList[1] < blk:
                        while (indexList[0] != rev[0]) and (indexList[1] != rev[1]): 
                            self.ACCoef[counter, chan] = block[indexList[0], indexList[1]]
                            print(counter, indexList[0], indexList[1], rev[0], rev[1])
                            counter += 1
                            indexList[1] -= 1
                            indexList[0] += 1

                        indexList[0] += 1
                        rev = indexList[::-1]

                        while ((indexList[0] != rev[0]) and (indexList[1] != rev[1])):
                            self.ACCoef[counter, chan] = block[indexList[0], indexList[1]]
                            print(counter, indexList[0], indexList[1])
                            counter += 1
                            indexList[1] += 1
                            indexList[0] -= 1
                        
                        indexList[1] += 1
                        rev = indexList[::-1]


    # This function has to seperate the transformed image into blocks and zig zag
    # Through it for the sake of encoding
    def zigZag(self):
        block = np.zeros(self.DCT.shape)
        blk = self.DCT.shape[0]
        imgH, imgW, numChan = self.quantisedImage.shape
        self.ACCoef = np.zeros(shape = ( (blk * blk) -1, 3))

        for chan in range(0, numChan):
            for row in range(0, imgH, blk):
                for col in range(0, imgW, blk):
                    block = self.quantisedImage[row:row+blk, col:col+blk, chan]
                    i = 0
                    j = 1
                    counter = 0
                    while j < blk and i < blk:
                        while j > 0:
                            self.ACCoef[counter, chan] = block[i, j]
                            print(counter, i, j)
                            counter += 1
                            j -= 1
                            i += 1
                            
                        if i < blk - 1:
                            self.ACCoef[counter, chan] = block[i, j]
                            print(counter, i, j)
                            counter += 1
                            i += 1 

                        if i == blk - 1:
                            j += 1
                            self.ACCoef[counter, chan] = block[i, j]
                            print(counter, i, j)
                            counter += 1
                            break
                        
                        while i > 0 and j < blk:
                            self.ACCoef[counter, chan] = block[i, j]
                            print(counter, i, j)
                            counter += 1
                            i -= 1
                            j += 1
                        
                        self.ACCoef[counter, chan] = block[i, j]
                        print(counter, i, j)
                        counter += 1
                        j += 1

                        if j == blk:
                            i += 1
                            self.ACCoef[counter, chan] = block[i, j]
                            print(counter, i, j)
                            counter += 1
                    i = 7
                    j = 1
                    while j < blk and i < blk:
                        while j < blk:
                            self.ACCoef[counter, chan] = block[i, j]
                            print(counter, i, j)
                            counter += 1
                            j += 1
                            i -= 1
                        j -= 1
                        i += 2

                        while i < blk:
                            self.ACCoef[counter, chan] = block[i, j]
                            print(counter, i, j)
                            counter += 1
                            j -= 1
                            i += 1
                        i -= 1
                        j += 2

     

        
encoder = Encoder()
encoder.constructDCT(8)
encoder.printDCT()
encoder.readFile('lena.png')
encoder.convertImageYUV()
encoder.applyDCT()
encoder.quantisation()
encoder.zag()
plt.show(encoder.imageEnc)
