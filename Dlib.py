

import cv2
import dlib
import time
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
    cv2.circle(image, (px, py), 1, (0, 255, 0), -1)

cap = cv2.VideoCapture(0)

cpt = 0
while(True):
	# Capture frame-by-frame
  ret, frame = cap.read()

    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=False, help="chemin de l'image initiale")
    # args = vars(ap.parse_args())
    # path_img=args["image"]

  # image=cv2.imread('data/Capture.JPG')

  if (cpt % 50 == 0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
    rects = detector(gray, 1)
    print('lol')

  for (i, rect) in enumerate(rects):
    landmarks = predictor(gray, rect)
    # draw_BB(rect,frame)
    draw_landmarks(landmarks,frame)

  if cv2.waitKey(1) & 0xFF == ord('a'):
    break

  cpt = cpt + 1
  print(cpt)
  cv2.imshow("Output", frame)


cap.release()
cv2.destroyAllWindows()




