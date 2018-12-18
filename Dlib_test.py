

import cv2
import dlib
import numpy as np
import argparse


def draw_BB(rect,image):
  x = rect.left()
  y = rect.top()
  w = rect.right() - x
  h = rect.bottom() - y
  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
def draw_landmarks(landmarks,image):
  for i, part in enumerate(landmarks.parts()):
    px = int(part.x)
    py = int(part.y)
    cv2.circle(image, (px, py), 1, (0, 0, 255), -1)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, help="chemin de l'image initiale")
args = vars(ap.parse_args())
path_img=args["image"]
 
image=cv2.imread(path_img)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


rects = detector(gray, 1)

for (i, rect) in enumerate(rects):
  landmarks = predictor(gray, rect)
  draw_BB(rect,image)
  draw_landmarks(landmarks,image)

  
cv2.imshow("Output", image)
cv2.waitKey(0)






