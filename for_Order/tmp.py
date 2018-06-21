import cv2

img = cv2.imread("lasso2.png")
cv2.imwrite('lasso_2.png', cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
