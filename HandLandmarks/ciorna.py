import cv2
import numpy as np
from matplotlib import pyplot as plt


def directional_image(gray_img):
    gaussian = cv2.GaussianBlur(gray_img, (3, 3), 0)
    sobel = cv2.Sobel(gaussian, cv2.CV_64F, 1, 1, ksize=3)
    return np.uint8(np.absolute(sobel))


def wrist_line_localization(gray_img):
    thresh = 127
    ret, thresh = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros(gray_img.shape)

    for c in contours:
        accuracy = 0.03 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, accuracy, True)
        cv2.drawContours(img_contours, [approx], 0, (255, 255, 0), 2)
        cv2.imshow('Approx Poly DP', img_contours)

    cv2.waitKey()


rgb_img = cv2.imread('img/hand2.jpg')
gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
wrist_line_localization(gray)
#
# plt.imshow(gray)
# plt.show()
# directional_img = directional_image(gray)
# plt.imshow(directional_img, cmap='gray')
# plt.show()

cv2.destroyAllWindows()