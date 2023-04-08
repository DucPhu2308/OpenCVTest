# import cv2
# thres = 0.45  # Threshold to detect object

# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)
# cap.set(10, 70)

# classNames = []
# classFile = 'coco.names'
# with open(classFile, 'rt') as f:
#     classNames = f.read().rstrip('\n').split('\n')

# configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# weightsPath = 'frozen_inference_graph.pb'

# net = cv2.dnn_DetectionModel(weightsPath, configPath)
# net.setInputSize(320, 320)
# net.setInputScale(1.0 / 127.5)
# net.setInputMean((127.5, 127.5, 127.5))
# net.setInputSwapRB(True)


# class Person:
#     def __init__(self, id, box) -> None:
#         self.id = id
#         self.box = box
#         self.backpackBox = None

# def isClose(box1, box2):
#     minDist = 80
#     return abs(box1[0] - box2[0]) < minDist and abs(box1[1] - box2[1]) < minDist

# Persons = []


# while True:
#     success, img = cap.read()
#     classIds, confs, bbox = net.detect(img, confThreshold=thres)
#     print(classIds, bbox)

#     match = False
#     if len(classIds) != 0:
#         detectResult = zip(classIds.flatten(), confs.flatten(), bbox)
#         if (len(Persons) == 0):
#             for classId, confidence, box in detectResult:
#                 if (classId == 1):
#                     if (len(Persons) == 0):
#                         id = 0
#                     else:
#                         id = Persons[-1].id + 1
#                     Persons.append(Person(id, box))
#         else:
#             # tim nguoi ko con ton tai
#             for person in Persons:
#                 for classId, confidence, box in detectResult:
#                     if (classId == 1):
#                         if isClose(box, person.box):
#                             person.box = box
#                             match = True
#                             break
#                 if not match:
#                     Persons.remove(person)
#                     match = False

#             # tim nguoi moi
#             for classId, confidence, box in detectResult:
#                 if (classId == 1):
#                     for person in Persons:
#                         if isClose(box, person.box):
#                             match = True
#                             break
#                     if not match:
#                         if (len(Persons) == 0):
#                             id = 0
#                         else:
#                             id = Persons[-1].id + 1
#                         Persons.append(Person(id, box))
#                         match = False
#                 if (classId == 27):  # backpack
#                     for person in Persons:
#                         if isClose(box, person.box):
#                             person.backpackBox = box
#                             break

#     for person in Persons:
#         cv2.rectangle(img, person.box, color=(0, 255, 0), thickness=2)
#         cv2.putText(img, "person " + str(person.id), (person.box[0]+10, person.box[1]+30),
#                     cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
#         if (person.backpackBox is not None):
#             cv2.rectangle(img, person.backpackBox,
#                           color=(0, 255, 0), thickness=2)
#             cv2.putText(img, "pack " + str(person.id), (person.backpackBox[0]+10, person.backpackBox[1]+30),
#                         cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

#             # if (classId == 1 or classId == 27):
#             #     cv2.rectangle(img,box,color=(0,255,0),thickness=2)
#             #     cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
#             #                     cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
#             #     cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
#             #     cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

#     cv2.imshow("Output", img)
#     cv2.waitKey(1)

import cv2
import random
import math


classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

thres = 0.6 # Threshold to detect object
CLASS_IDS = {
    'person': 1,
    'bag': 27
}

class Object:
    def isClose(self, box):
        minDist = 80

        x1 = self.box[0] + self.box[2] / 2
        y1 = self.box[1] + self.box[3] / 2

        x2 = box[0] + box[2] / 2
        y2 = box[1] + box[3] / 2

        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2) <= minDist 

class Person(Object):
    def __init__(self, confidence, box):
        self.id = str(random.randint(1000, 9999))
        self.confidence = confidence
        self.box = box


class Bag(Object):
    def __init__(self, confidence, box):
        self.id = random.randint(1000, 9999)
        self.confidence = confidence
        self.box = box

        self.owner_id = 'None'

    def isVeryClose(self, person):
        minDist = 100

        x1 = self.box[0] + self.box[2] / 2
        y1 = self.box[1] + self.box[3] / 2

        x2 = person.box[0] + person.box[2] / 2
        y2 = person.box[1] + person.box[3] / 2

        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2) <= minDist 


class Objects:
    def __init__(self):
        self.people = []
        self.bags = []

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3,1280)
        self.cap.set(4,720)
        self.cap.set(10,70)

        self.net = cv2.dnn_DetectionModel(weightsPath,configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/ 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

    def detect(self):
        success, self.img = self.cap.read()
        classIds, confs, bbox = self.net.detect(self.img,confThreshold=thres)

        new_people = []
        new_bags = []
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            if classId == CLASS_IDS['person']:
                existed = False 
                for person in self.people:
                    if person.isClose(box):
                        existed = True 
                        new_people.append(person)
                        break

                if not existed:
                    new_people.append(Person(confidence, box))

            elif classId == CLASS_IDS['bag']:
                existed = False 
                for bag in self.bags:
                    if bag.isClose(box):
                        # Find owner
                        for person in self.people:
                            if bag.isVeryClose(person):
                                bag.owner_id = person.id
                        existed = True 
                        new_bags.append(bag)
                        break

                if not existed:
                    new_bags.append(Bag(confidence, box))

                

        self.people = new_people
        self.bags = new_bags
        print(self.people, new_people)

    def draw(self):
        for person in self.people:
            cv2.rectangle(self.img,person.box,color=(0,255,0),thickness=2)
            cv2.putText(self.img,'Person ' + person.id,(person.box[0]+10,person.box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            # cv2.putText(self.img,str(round(person.confidence*100,2)),(person.box[0]+200,person.box[1]+30),
            # cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        for bag in self.bags:
            cv2.rectangle(self.img,bag.box,color=(0,255,0),thickness=2)
            cv2.putText(self.img,'Bag ' + bag.owner_id,(bag.box[0]+10,bag.box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
            # cv2.putText(self.img,str(round(bag.confidence*100,2)),(bag.box[0]+200,bag.box[1]+30),
            # cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        cv2.imshow('Output', self.img)
        cv2.waitKey(1)
        
            

objects = Objects()
while True:
    objects.detect()
    objects.draw()

    
    