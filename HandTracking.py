import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
handDetector = mpHands.Hands()

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = handDetector.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handLms)

    cv2.imshow("Image",img)
    cv2.waitKey(1)