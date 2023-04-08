import cv2
from object import *

classNames = []
def loadClassName():
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

#  Configurations
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
thres = 0.45 # Threshold to detect object

# Constant
CLASS_IDS = {
    'PERSON': 1,
    'BAG': 77
}

# for i, name in enumerate(classNames):
#     CLASS_IDS[classNames[i].upper()] = i + 1

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

    def detectObjects(self):
        success, self.img = self.cap.read()
        classIds, confs, bbox = self.net.detect(self.img, confThreshold=thres)
        new_people, new_bags = [], []
        if len(classIds) == 0:
            self.people, self.bags = [], []
            return

        
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classId == CLASS_IDS["PERSON"]:
                new_person = Person(confidence, box)
                for person in self.people:
                    if new_person.isVeryClose(person.box):
                        new_person.id = person.id 
                        break

                new_people.append(new_person)

            elif classId == CLASS_IDS["BAG"]:
                new_bag = Bag(confidence, box)
                # print(len(self.bags))
                for bag in self.bags:
                    if bag.isVeryClose(box):
                        for person in self.people:
                            if bag.isClose(person) and bag.owner_id == 'None':
                                bag.owner_id = person.id
                                person.object_ids.append(bag.id)
                        new_bag.id, new_bag.owner_id = bag.id, bag.owner_id  
                        break

                for new_bag in new_bags:
                    # Find owner
                    if new_bag.owner_id == 'None':
                        for person in self.people:
                            if new_bag.isClose(person):
                                new_bag.setOwner(person.id)
                                person.object_ids.append(new_bag.id)

                for bag in self.bags:
                    print(bag.owner_id, end=' ')
                new_bags.append(new_bag)
        
        self.people, self.bags = new_people, new_bags

    def check(self):
        for person in self.people:
            print(person.object_ids)
            # Check if each person is controlling enough their belongings
            for id in person.object_ids:
                exist = False
                for bag in self.bags:
                    if bag.id == id:
                        exist = True 
                        break 
                if not exist:
                    print('Person ' + person.id + ' losed control a belonging')
                    break
                    

            left_objects = person.object_ids
            for bag in self.bags:
                if bag.owner_id == person.id and person.calDistance(bag.box) >= 400:
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
            # print(bag.owner_id)
            cv2.rectangle(self.img, bag.box, color=(0, 255, 0), thickness=2)
            cv2.putText(self.img, 'Phone ' + bag.owner_id, (bag.box[0]+10, bag.box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
            # cv2.putText(self.img,str(round(bag.confidence*100,2)),(bag.box[0]+200,bag.box[1]+30),
            # cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        cv2.imshow('Object Detector', self.img)
        cv2.waitKey(1)
