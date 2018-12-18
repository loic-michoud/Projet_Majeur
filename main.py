import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour

#LBP is faster (a few times faster) but less accurate. (10-20% less than Haar).
# If you want to detect faces on an embedded system, I think LBP is the choice,
# because it does all the calculations in integers. Haar uses floats, which is a killer for embedded/mobile.

cap = cv2.VideoCapture(0)

# Create the haar or lbp cascade
#faceCascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
faceCascade = cv2.CascadeClassifier("data/lbpcascade_frontalface.xml")

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
		#flags = cv2.CV_HAAR_SCALE_IMAGE
	)

	print("Found {0} faces!".format(len(faces)))

	# a=[]
	# b=[]
	s = np.linspace(0, 2*np.pi, 400)
	a = 190 + 150*np.cos(s)
	b = 320 + 150*np.sin(s)
	init = np.array([a, b]).T
	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
	    cv2.ellipse(frame,(x+w/2,y+h/2),(w,h),0,0,360,(0,255,0))
	    # a.append(x)
	    # b.append(y)
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
	# snake=active_contour(gaussian(frame, 3),
                       # init, alpha=0.015, beta=10, gamma=0.001)

	# Display the resulting frame
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('a'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()