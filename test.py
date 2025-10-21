import argparse
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

assert insightface.__version__>='0.3'

parser = argparse.ArgumentParser(description='insightface app test')
# general
parser.add_argument('--ctx', default=0, type=int, help='ctx id, <0 means using cpu')
parser.add_argument('--det-size', default=640, type=int, help='detection size')
args = parser.parse_args()

app = FaceAnalysis()
app.prepare(ctx_id=args.ctx, det_size=(args.det_size,args.det_size))

img = ins_get_image('t1')
faces = app.get(img)
assert len(faces)==6
rimg = app.draw_on(img, faces)
cv2.imwrite("t1_output.jpg", rimg)

img = cv2.imread('porkys.jpg')
faces2 = app.get(img)
assert len(faces)==6
rimg = app.draw_on(img, faces2)
cv2.imwrite("t2_output.jpg", rimg)
# then print all-to-all face similarity
feats,whois = [],[]
for face in faces:
    feats.append(face.normed_embedding)
    whois.append(face.sex+str(face.age))
feats2 = []
for face in faces2:
    feats2.append(face.normed_embedding)
feats = np.array(feats, dtype=np.float32)
feats2 = np.array(feats2, dtype=np.float32)

# sims = np.dot(feats, feats2.T)
# np.set_printoptions(suppress=True,
#    formatter={'float_kind':'{:0.2f}'.format})  
# print(sims)
# print(type(sims))
# print(np.argwhere(sims > 0.4))
# print("Values bigger than 10 =", len(sims[sims>0.4]))
# print("Their indices are ", np.nonzero(sims > 0.4))
for i in range(len(faces)):
    print(whois[i],'-',end='')
    print('%1d  ' % i, end='')
    for j in range(len(faces2)):
        dist = np.sqrt(np.sum(np.square(np.subtract(feats[i,:], feats2[j,:]))))
        if dist < 1.1:
            print('Match!')
            next
        print('  %1.4f  ' % dist, end='')
    print('')