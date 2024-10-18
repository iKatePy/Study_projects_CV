from math import fabs
import numpy as np
import cv2
from matplotlib import pyplot

imgR = cv2.imread('stereoL.png', 0)
imgL = cv2.imread('stereoR.png', 0)

stereo = cv2.StereoBM_create(numDisparities = 16, blockSize = 23)
disparity = stereo.compute(imgL,imgR)

pyplot.figure(figsize=(5,5))
pyplot.imshow(disparity, 'gray')
pyplot.xticks([])
pyplot.yticks([])
pyplot.show()