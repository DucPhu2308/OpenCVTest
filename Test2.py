import object_detector as od
import cv2

objects = od.ObjectDetector()
while True:
    objects.detectObjects()
    objects.check()
    objects.draw()  
    cv2.waitKey(1)
