from object import *
import random
import math

class Object:
    def isVeryClose(self, box):
        minDist = 70
        return self.calDistance(box) <= minDist

    def calDistance(self, box):
        x1 = self.box[0] + self.box[2] / 2
        y1 = self.box[1] + self.box[3] / 2

        x2 = box[0] + box[2] / 2
        y2 = box[1] + box[3] / 2

        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


class Person(Object):
    def __init__(self, confidence, box):
        self.id = str(random.randint(1000, 9999))
        self.confidence = confidence
        self.box = box
        self.object_ids = []


class Bag(Object):
    def __init__(self, confidence, box):
        self.id = str(random.randint(1000, 9999))
        self.confidence = confidence
        self.box = box

        self.owner_id = 'None'

    def setOwner(self, person_id):
        self.owner_id = person_id

    def isClose(self, person):
        minDist = 300
        return self.calDistance(person.box) <= minDist
