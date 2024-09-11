import cv2
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(4)

img = cv2.imread('quan.jpg')
img = cv2.resize(img, (200, 250))
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255


class Conv2D:
    def __init__(self, input, numOfKernel, kernelSize,padding, stride):
        self.input = np.pad(input, ((padding, padding), (padding, padding)), 'constant')
        self.stride = stride
        self.kernel = np.random.randn(numOfKernel, kernelSize, kernelSize)
        self.result = np.zeros((int((self.input.shape[0] - self.kernel.shape[1])/self.stride) + 1,
                                int((self.input.shape[1] - self.kernel.shape[2])/self.stride) + 1, self.kernel.shape[0]))
        
    # roi: region of interesting    
    def getROI(self):
        for row in range(0, int((self.input.shape[0] - self.kernel.shape[1])/self.stride) + 1):
            for col in range(0, int((self.input.shape[1] - self.kernel.shape[2])/self.stride) + 1):
                roi = np.sum(self.input[row*self.stride : row*self.stride + self.kernel.shape[1],
                                        col*self.stride : col*self.stride + self.kernel.shape[2]] * self.kernel)
                yield row, col, roi
                # yield -> trả về trong vòng lặp | return trả về chắc lần lặp đầu 
    
    def operate(self):
        for layer in range(self.kernel.shape[0]):
            for row, col, roi in self.getROI():
                self.result[row , col, layer] = np.sum(roi * self.kernel[layer]) 

        return self.result

class ReLU:
    def __init__(self, input):
        self.input = input
        self.result = np.zeros(self.input.shape)
    
    def operate(self):
        for layer in range (self.input.shape[2]):
            for row in range(self.input.shape[0]):
                for col in range(self.input.shape[1]):
                    self.result[row, col, layer] = 0 if self.input[row, col, layer] < 0 else self.input[row, col, layer]
        return self.result

class LeakyReLU:
    def __init__(self, input, alpha=0.01):
        self.input = input
        self.alpha = alpha
        self.result = np.zeros(self.input.shape)

    def operate(self):
        for layer in range(self.input.shape[2]):
            for row in range(self.input.shape[0]):
                for col in range(self.input.shape[1]):
                    if self.input[row, col, layer] < 0:
                        self.result[row, col, layer] = self.alpha * self.input[row, col, layer]
                    else:
                        self.result[row, col, layer] = self.input[row, col, layer]
        return self.result

class MaxPooling:
    def __init__(self, input, poolingSize=2):
        self.input = input
        self.poolingSize = poolingSize
        self.result = np.zeros((int(self.input.shape[0]/self.poolingSize),
                                int(self.input.shape[1]/self.poolingSize),
                                self.input.shape[2]))

    def operate(self):
        for layer in range(self.input.shape[2]):
            for row in range(int(self.input.shape[0]/self.poolingSize)):
                for col in range(int(self.input.shape[1]/self.poolingSize)):
                    self.result[row, col, layer] = np.max(self.input[row*self.poolingSize : row*self.poolingSize + self.poolingSize,
                                                                     col*self.poolingSize : col*self.poolingSize + self.poolingSize,
                                                                     layer])
        return self.result

class SoftMax:
    def __init__(self, input, nodes):
            self.input = input
            self.nodes = nodes
            # y = w0 + w(i)*x(i)
            self.flatten = self.input.flatten()
            self.weight = np.random.randn(self.flatten.shape[0])/self.flatten.shape[0]
            self.bias = np.random.randn(self.nodes)

    def operate(self):
        totals = np.dot(self.flatten, self.weight) + self.bias
        probabilities = np.exp(totals) / np.sum(np.exp(totals))
        return probabilities


conv2D = Conv2D(gray_img, 16, 3, 0, 1)
img_gray_conv2d = conv2D.operate()
reLU_image = ReLU(img_gray_conv2d).operate()
leakyReLU_image = LeakyReLU(img_gray_conv2d).operate()
maxPooling_image = MaxPooling(leakyReLU_image, 3).operate()
softMax_image = SoftMax(maxPooling_image, 10).operate()
print(softMax_image)

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(leakyReLU_image[:, :, i], cmap='gray')
    plt.axis('off')

plt.savefig('quan_lkReLU.jpg')
plt.show()
