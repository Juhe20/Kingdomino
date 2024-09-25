import cv2
import numpy as np


#import billeder
img = cv2.imread("1.jpg")
imgHvid = img.copy()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hvid = cv2.imread('hvid.png')
Grayimg = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)
Crown_N = cv2.imread("Crown_N.png")
Crown_S = cv2.imread("Crown_S.png")
Crown_E = cv2.imread("Crown_E.png")
Crown_W = cv2.imread("Crown_W.png")

template_height, template_width = Crown_N.shape[:2]
template_height1, template_width1 = Crown_S.shape[:2]
template_height2, template_width2 = Crown_E.shape[:2]
template_height3, template_width3 = Crown_W.shape[:2]

matchresult1 = cv2.matchTemplate(img, Crown_N, cv2.TM_CCOEFF_NORMED)
matchresult2 = cv2.matchTemplate(img, Crown_S, cv2.TM_CCOEFF_NORMED)
matchresult3 = cv2.matchTemplate(img, Crown_E, cv2.TM_CCOEFF_NORMED)
matchresult4 = cv2.matchTemplate(img, Crown_W, cv2.TM_CCOEFF_NORMED)

def drawcircles(threshold, matchresult):
    count = 0
    result = matchresult
    yloc, xloc = np.where(result >= threshold)
    for (x, y) in zip(xloc, yloc):
        cv2.circle(img, (x + template_width // 2, y + template_height // 2), 5, (255, 255, 255), -1)
        count += 1

drawcircles(0.7, matchresult1)
drawcircles(0.73, matchresult2)
drawcircles(0.9, matchresult3)
drawcircles(0.687, matchresult4)

finalRes = cv2.matchTemplate(img, hvid, cv2.TM_CCOEFF_NORMED)

cv2.imshow('Detected crowns', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


