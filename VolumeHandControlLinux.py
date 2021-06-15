import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from subprocess import call

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
cTime = 0

detector = htm.handDetector(maxHands=1)

minVol = 0
maxVol = 100
volBar = 400
volPer = 0
area = 0



while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)
    if len(lmList) != 0:

        area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])//100
        #print(area)

        if 200 < area < 1000:
            length, img, lineInfo = detector.findDistance(4, 8, img)
            # print(length)

            # My hand range: 20 - 250
            # Volume range -65 - 0

            vol = np.interp(length, [20, 200], [minVol, maxVol])
            volBar = np.interp(length, [20, 200], [400, 150])
            volPer = np.interp(length, [20, 200], [0, 100])
            
            smoothness = 10
            volPer = smoothness * round(volPer/smoothness)

            fingers = detector.fingersUp()
            # print(fingers)
            
            if not fingers[3]:
                call(["amixer", "-D", "pulse", "-q", "sset", "Master", str(vol)+"%"])
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv2.FILLED)

        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(volPer)), (45, 450), cv2.FONT_ITALIC, 1, (0, 250, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)