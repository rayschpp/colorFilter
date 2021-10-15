# Source: https://medium.com/featurepreneur/colour-filtering-and-colour-pop-effects-using-opencv-python-3ce7d4576140
#import the libraries
import cv2 as cv
import numpy as np
import time

from app import app

#read the image
img = cv.imread("image.jpg")
#convert the BGR image to HSV colour space
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#obtain the grayscale image of the original image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#set the bounds for the red hue
lower_red_1 = np.array([0,100,20])
upper_red_1 = np.array([10,255,255])
lower_red_2 = np.array([160, 100, 20])
upper_red_2 = np.array([180, 255, 255])

# set bounds for blue hue
lower_blue = np.array([100,150,0])
upper_blue = np.array([120,255,255])

# set bounds for yellow hue
lower_orange = np.array([15, 100, 50])
upper_orange = np.array([30, 255, 255])

#create a mask using the bounds set
red_lower_mask = cv.inRange(hsv, lower_red_1, upper_red_1)
red_upper_mask = cv.inRange(hsv, lower_red_2, upper_red_2)

blue_mask = cv.inRange(hsv, lower_blue, upper_blue)


# orange mask
orange_mask = cv.inRange(hsv, lower_orange, upper_orange)

full_mask = orange_mask

#create an inverse of the mask
mask_inv = cv.bitwise_not(full_mask)
#Filter only the red colour from the original image using the mask(foreground)
res = cv.bitwise_and(img, img, mask=full_mask)
#Filter the regions containing colours other than red from the grayscale image(background)
background = cv.bitwise_and(gray, gray, mask = mask_inv)
#convert the one channelled grayscale background to a three channelled image
background = np.stack((background,)*3, axis=-1)
#add the foreground and the background
added_img = cv.add(res, background)

cv.imwrite(time.strftime("%Y%m%d-%H%M%S") + ".png", added_img)

#create resizable windows for the images
cv.namedWindow("res", cv.WINDOW_NORMAL)
cv.namedWindow("hsv", cv.WINDOW_NORMAL)
cv.namedWindow("mask", cv.WINDOW_NORMAL)
cv.namedWindow("added", cv.WINDOW_NORMAL)
cv.namedWindow("back", cv.WINDOW_NORMAL)
cv.namedWindow("mask_inv", cv.WINDOW_NORMAL)
cv.namedWindow("gray", cv.WINDOW_NORMAL)

#display the images
cv.imshow("back", background)
cv.imshow("mask_inv", mask_inv)
cv.imshow("added",added_img)
cv.imshow("mask", full_mask)
cv.imshow("gray", gray)
cv.imshow("hsv", hsv)
cv.imshow("res", res)

if cv.waitKey(0):
    cv.destroyAllWindows()

