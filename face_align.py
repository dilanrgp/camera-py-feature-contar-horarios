from mtcnn import MTCNN
import cv2
import time

img = cv2.cvtColor(cv2.imread("G://Film/3/IMG_0001.JPG"), cv2.COLOR_BGR2RGB)

st = time.time()
detector = MTCNN()

result = detector.detect_faces(img)
et = time.time()

print(str(et - st))

print(result)
