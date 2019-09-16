import numpy as np
import cv2 as cv
import imutils
import os
from imutils.face_utils import FaceAligner
from dlib import shape_predictor
from dlib import rectangle


class Aligner:

    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'

        proto_path = dir_path + 'face_detection_model/deploy.prototxt'
        model_path = dir_path + 'face_detection_model/res10_300x300_ssd_iter_140000.caffemodel'

        self.detector = cv.dnn.readNetFromCaffe(proto_path, model_path)

        predictor_path = dir_path + "face_detection_model/shape_predictor_68_face_landmarks.dat"

        self.predictor = shape_predictor(predictor_path)
        self.fa = FaceAligner(self.predictor, desiredFaceWidth=256, desiredLeftEye=(.38, .38))

    def align_image_get_boxes(self, image):
        frame = image

        image = imutils.resize(frame, width=600)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        image_blob = cv.dnn.blobFromImage(image, 1.0, (400, 400), (104.0, 177.0, 123.0), swapRB=False, crop=False)

        height, width = image.shape[:2]
        self.detector.setInput(image_blob)
        rectangles = self.detector.forward()
        boxes = []
        faces = []
        for x in range(0, rectangles.shape[2]):
            confidence = rectangles[0, 0, x, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > .6:
                box = rectangles[0, 0, x, 3:7] * np.array([width, height, width, height])
                boxes.append(box)
                (start_x, start_y, end_x, end_y) = box.astype("int")
                rect = rectangle(start_x, start_y, end_x, end_y)
                face_aligned = self.fa.align(image, gray, rect)
                faces.append(face_aligned)
        return faces, boxes

    def get_boxes(self, image):
        frame = image
        image = imutils.resize(frame, width=600)
        image_blob = cv.dnn.blobFromImage(image, 1.0, (400, 400), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        height, width = image.shape[:2]
        self.detector.setInput(image_blob)
        rectangles = self.detector.forward()
        boxes = []
        for x in range(0, rectangles.shape[2]):
            confidence = rectangles[0, 0, x, 2]
            if confidence > .6:
                box = rectangles[0, 0, x, 3:7] * np.array([width, height, width, height])
                boxes.append(box)
        return boxes

    def get_center_face_box(self, image):
        frame = image

        image = imutils.resize(frame, width=600)

        image_blob = cv.dnn.blobFromImage(image, 1.0, (400, 400), (104.0, 177.0, 123.0), swapRB=False, crop=False)

        height, width = image.shape[:2]
        true_x_center = int(width / 2)
        true_y_center = int(height / 2)
        self.detector.setInput(image_blob)
        rectangles = self.detector.forward()
        boxes = []
        distances = []
        for x in range(0, rectangles.shape[2]):
            confidence = rectangles[0, 0, x, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > .6:
                box = rectangles[0, 0, x, 3:7] * np.array([width, height, width, height])
                boxes.append(box)
                (start_x, start_y, end_x, end_y) = box.astype("int")
                x_center = start_x + int((end_x - start_x) / 2)
                y_center = start_y + int((end_y - start_y) / 2)
                x_distance = true_x_center - x_center
                y_distance = true_y_center - y_center
                true_distance = pow(pow(x_distance, 2) + pow(y_distance, 2), 1/2)       # sqrt(x^2 + y^2)
                distances.append(true_distance)
        min_distance = np.inf
        center_index = 0
        for i, distance in enumerate(distances):
            if distance < min_distance:
                min_distance = distance
                center_index = i

        return center_index, boxes

    def align_image(self, image):
        frame = image

        image = imutils.resize(frame, width=600)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        image_blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0),
                                          swapRB=False, crop=False)
        height, width = image.shape[:2]
        self.detector.setInput(image_blob)
        rectangles = self.detector.forward()
        if len(rectangles) > 0:
            # Get most likely face
            i = np.argmax(rectangles[0, 0, :, 2])
            box = rectangles[0, 0, i, 3:7] * np.array([width, height, width, height])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            rect = rectangle(start_x, start_y, end_x, end_y)
            face_aligned = self.fa.align(image, gray, rect)
            return face_aligned
        else:
            print("Could not find a face in this image")
            return None
