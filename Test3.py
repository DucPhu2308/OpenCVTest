import cv2
import math
thres = 0.5  # Threshold to detect object

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


class Person:
    def __init__(self, id, box) -> None:
        self.id = id
        self.box = box
        self.backpackBox = None

def isClose(box1, box2, minDist):
    return calDistance(box1,box2) < minDist

def calDistance(box1, box2):
    return math.sqrt((getCenter(box1)["x"] - getCenter(box2)["x"])**2 + (getCenter(box1)["y"] - getCenter(box2)["y"])**2)

def getCenter(box):
    return {"x": int(box[0] + box[2]/2), 
            "y": int(box[1] + box[3]/2)}

Persons = []


while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    # print(classIds, bbox)

    match = False
    if len(classIds) != 0:
        detectResult = zip(classIds.flatten(), confs.flatten(), bbox)
        if (len(Persons) == 0):
            for classId, confidence, box in detectResult:
                if (classId == 1):
                    if (len(Persons) == 0):
                        id = 0
                    else:
                        id = Persons[-1].id + 1
                    Persons.append(Person(id, box))
        else:
            # tim nguoi ko con ton tai
            for person in Persons:
                for classId, confidence, box in detectResult:
                    if (classId == 1):
                        if isClose(box, person.box, 100):
                            person.box = box
                            match = True
                            break
                if not match:
                    Persons.remove(person)
                    match = False

            # tim nguoi moi
            for classId, confidence, box in detectResult:
                if (classId == 1):
                    for person in Persons:
                        if isClose(box, person.box,100):
                            match = True
                            break
                    if not match:
                        if (len(Persons) == 0):
                            id = 0
                        else:
                            id = Persons[-1].id + 1
                        Persons.append(Person(id, box))
                        match = False
                if (classId == 27):  # backpack
                    for person in Persons:
                        if (person.backpackBox is not None and isClose(box, person.backpackBox,80))or (person.backpackBox is None and isClose(box, person.box,80)):
                            person.backpackBox = box
                            break

    for person in Persons:
        cv2.rectangle(img, person.box, color=(0, 255, 0), thickness=2)
        cv2.circle(img, (getCenter(person.box)["x"], getCenter(person.box)["y"]), 10, color=(0, 255, 0), thickness=2)
        cv2.putText(img, "person " + str(person.id), (person.box[0]+10, person.box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        if (person.backpackBox is not None):
            cv2.rectangle(img, person.backpackBox, color=(0, 255, 0), thickness=2)
            cv2.putText(img, "pack " + str(person.id), (person.backpackBox[0]+10, person.backpackBox[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            
            if (calDistance(person.box, person.backpackBox)) > 200:
                print("Alert!")




            # if (classId == 1 or classId == 27):
            #     cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            #     cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
            #                     cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            #     cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
            #     cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow("Output", img)
    cv2.waitKey(1)