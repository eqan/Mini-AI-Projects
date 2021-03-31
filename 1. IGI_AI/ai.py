'''
    * Current Features:
        * Human Detection
        * Human Kill
    * TODO:
        * Integerate Mouse Movement To Target
        * Integerate Deep Learning For More Accurate Human Detection
        * Player Movement Controls By AI
        * Teach The Player How To Complete A Mission
        * Bring Up FrameRates
'''
import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
from mss import mss
import time
import pydirectinput

bounding_box = {'top': 40, 'left': 64, 'width': 622, 'height': 425}
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

sct = mss()

def detect_and_kill(frame):
    person = 0
    # Method 1 Human Detection Technique
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(gray, winStride = (8, 8), padding = (16, 16), scale = 1.03, useMeanshiftGrouping=False, finalThreshold=0.2)

    #Converting coordinates from lists to list
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    # Supression for multiple boxes generated on one object, thus in the end converted into a single  object
    boxes = non_max_suppression(boxes, probs=None, overlapThresh=0.9)
    if(len(boxes) > 0):
	    # Count number of persons
	    for i, (x, y, w, h) in enumerate(boxes):
	        # display the detected boxes in the colour picture
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                pydirectinput.move(int(x/4), int(y/4))
                person += 1
    if(person > 0):
        pydirectinput.press('f')
    cv2.putText(frame, f'Total Persons : {person}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.imshow('OpenCV Screen', frame)

time.sleep(15)
while True:
    if(pydirectinput.press('s')):
        time.sleep(15)
    frame = np.array(sct.grab(bounding_box))
    detect_and_kill(frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
    	cv2.destroyAllWindows()
    	break
