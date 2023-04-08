import cv2
import random
import math


classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

thres = 0.45 # Threshold to detect object
CLASS_IDS = {
    'person': 1,
    'bag': 27
}


class Object:
    def isClose(self, box):
        minDist = 120
        self.countDistance(box) <= minDist

    def countDistance(self, box):
        x1 = self.box[0] + self.box[2] / 2
        y1 = self.box[1] + self.box[3] / 2

        x2 = box[0] + box[2] / 2
        y2 = box[1] + box[3] / 2

        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


class Person(Object):
    def __init__(self, confidence, box, id='random'):
        if id == 'random':
            self.id = str(random.randint(1000, 9999))
        self.confidence = confidence
        self.box = box
        self.object_ids = []


class Bag(Object):
    def __init__(self, confidence, box):
        self.id = random.randint(1000, 9999)
        self.confidence = confidence
        self.box = box

        self.owner_id = 'None'

    def isOwned(self, person):
        minDist = 200

        x1 = self.box[0] + self.box[2] / 2
        y1 = self.box[1] + self.box[3] / 2

        x2 = person.box[0] + person.box[2] / 2
        y2 = person.box[1] + person.box[3] / 2

        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2) <= minDist


class ObjectDetector:
    def __init__(self):
        self.people = []
        self.bags = []

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.cap.set(10, 70)

        self.net = cv2.dnn_DetectionModel(weightsPath, configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

    def detect(self):
        success, self.img = self.cap.read()
        classIds, confs, bbox = self.net.detect(self.img, confThreshold=thres)
        if len(classIds) == 0:
            return

        new_people = []
        new_bags = []
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classId == CLASS_IDS['person']:
                existed = False
                for person in self.people:
                    if person.isClose(box):
                        existed = True
                        new_people.append(Person(confidence, box, person.id))
                        break

                if not existed:
                    new_people.append(Person(confidence, box))

            elif classId == CLASS_IDS['bag']:
                existed = False
                for bag in self.bags:
                    if bag.isClose(box):
                        # Find owner
                        for person in self.people:
                            if bag.isOwned(person) and bag.owner_id == 'None':
                                bag.owner_id = person.id
                                person.object_ids.append(bag.id)
                        existed = True
                        new_bags.append(bag)
                        break

                if not existed:
                    new_bags.append(Bag(confidence, box))

        self.people = new_people
        self.bags = new_bags

    def check(self):
        for person in self.people:
            left_objects = person.object_ids
            for bag in bags:
                if bag.owner_id == person.id and person.countDistance(bag.box):
                    left_objects.remove(bag.id)

            if len(left_objects) != 0:
                print('Left')

    def draw(self):
        for person in self.people:
            cv2.rectangle(self.img, person.box, color=(0, 255, 0), thickness=2)
            cv2.putText(self.img, 'Person ' + person.id, (person.box[0]+10, person.box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(self.img,str(round(person.confidence*100,2)),(person.box[0]+200,person.box[1]+30),
            # cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        for bag in self.bags:
            cv2.rectangle(self.img, bag.box, color=(0, 255, 0), thickness=2)
            cv2.putText(self.img, 'Bag ' + bag.owner_id, (bag.box[0]+10, bag.box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
            # cv2.putText(self.img,str(round(bag.confidence*100,2)),(bag.box[0]+200,bag.box[1]+30),
            # cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        cv2.imshow('Output', self.img)
        cv2.waitKey(1)


objects = ObjectDetector()
while True:
    objects.detect()
    objects.draw()
