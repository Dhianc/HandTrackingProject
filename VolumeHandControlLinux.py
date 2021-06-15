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

detector = htm.handDetector()

minVol = 0
maxVol = 100
volBar = 400
volPer = 0


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2) // 2, (y1+y2) // 2

        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)

        # My hand range: 20 - 250
        # Volume range -65 - 0

        vol = np.interp(length, [20, 200], [minVol, maxVol])
        volBar = np.interp(length, [20, 200], [400, 150])
        volPer = np.interp(length, [20, 200], [0, 100])
        print(length, vol)
        if (vol <= 100) and (vol >= 0):
            call(["amixer", "-D", "pulse", "sset", "Master", str(vol)+"%"])
            valid = True

        if length < 20:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(volPer)), (45, 450), cv2.FONT_ITALIC, 1, (0, 250, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_ITALIC, 1, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)