import cv2
import os
import shutil

filepath1 = './c4_new/2019-9-2-11-37-25.jpg'
filepath2 = './c4_new/2019-9-2-11-33-2.jpg'


import cv2

# 读入图像
img = cv2.imread(filepath1)

# 加载人脸特征，该文件在 python安装目录\Lib\site-packages\cv2\data 下

face_cascade = cv2.CascadeClassifier(
    r'C:\\Program Files\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')

# 将读取的图像转为COLOR_BGR2GRAY，减少计算强度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 检测出的人脸个数
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(5, 5))

print("Face : {0}".format(len(faces)))

# 用矩形圈出人脸的位置
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.namedWindow("Faces")
cv2.imshow("Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(1)
# 读入图像
for file in os.listdir('./c4'):
    filepath1 = os.path.join('./c4', file)
    img = cv2.imread(filepath1)
    face_cascade = cv2.CascadeClassifier(
        r'C:\\Program Files\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(5, 5))
    if str(type(faces)) != "<class 'tuple'>":
        shutil.copy(filepath1, os.path.join('./c4_new', file))
        print(filepath1)
