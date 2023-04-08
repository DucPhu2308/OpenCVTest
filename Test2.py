import object_detector as od

objects = od.ObjectDetector()
while True:
    objects.detectObjects()
    objects.check()
    objects.draw()
