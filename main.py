import cv2
import numpy as np

#import billeder
img = cv2.imread("1.jpg")
imgHvid = img.copy()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


Grayimg = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)

Crown_N = cv2.imread("Crown_N.png")
Crown_S = cv2.imread("Crown_S.png")
Crown_E = cv2.imread("Crown_E.png")
Crown_W = cv2.imread("Crown_W.png")


template_height, template_width = Crown_N.shape[:2]
template_height1, template_width1 = Crown_S.shape[:2]
template_height2, template_width2 = Crown_E.shape[:2]
template_height3, template_width3 = Crown_W.shape[:2]




result = cv2.matchTemplate(img, Crown_N, cv2.TM_CCOEFF_NORMED)
result1 = cv2.matchTemplate(img, Crown_S, cv2.TM_CCOEFF_NORMED)
result2 = cv2.matchTemplate(img, Crown_E, cv2.TM_CCOEFF_NORMED)
result3 = cv2.matchTemplate(img, Crown_W, cv2.TM_CCOEFF_NORMED)



threshold_N = 0.7# Adjust this value based on your needs
yloc_N, xloc_N = np.where(result >= threshold_N)

threshold_S = 0.73
yloc_S, xloc_S = np.where(result1 >= threshold_S)

threshold_E = 0.9
yloc_E, xloc_E = np.where(result2 >= threshold_E)

threshold_W = 0.687
yloc_W, xloc_W = np.where(result3 >= threshold_W)

# Assuming xloc_N and yloc_N are defined and img, template_width, template_height are initialized
xPos = []
yPos = []
count = 0

for (x, y) in zip(xloc_N, yloc_N):
    xPos.append(x)
    yPos.append(y)
    cv2.circle(img, (x + template_width // 2, y + template_height // 2), 5, (255, 255, 255), -1)
    print(x, y)

for i in range(len(xPos)):
    if not (xPos[i] - 2 <= x <= xPos[i] +2 or yPos[i] - 2 <= y <= yPos[i] + 2):
        count += 1
print(count)
for (x, y) in zip(xloc_S, yloc_S):
    cv2.circle(img, (x + template_width1 // 2, y + template_height1 // 2), 5, (255, 255, 255), -1)


for (x, y) in zip(xloc_E, yloc_E):
    cv2.circle(img, (x + template_width2 // 2, y + template_height2 // 2), 5, (255, 255, 255), -1)


for (x, y) in zip(xloc_W, yloc_W):
    cv2.circle(img, (x + template_width3 // 2, y + template_height3 // 2), 5, (255, 255, 255), -1)



cv2.imshow('Detected Hearts', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

