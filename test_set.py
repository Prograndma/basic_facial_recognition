import os
from person import Person
import numpy as np
import time
from imutils import paths
from face_aligner import Aligner
import cv2 as cv


dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'

embedding_tool = cv.dnn.readNetFromTorch(f"{dir_path}nn4.small2.v1.t7")


class TestSet:
    def __init__(self, name, directory='test/'):
        self.name = name
        self.directory = directory
        self.person_list = []

    def add_person_to_list(self, person):
        if isinstance(self.person_list, list):
            self.person_list.append(person)
        else:
            self.person_list = np.append(self.person_list, person)
        # self.save_embeddings()

    def has_person_named(self, name):
        for person in self.person_list:
            if person.name == name:
                return True
        return False

    def get_person_named(self, name):
        for person in self.person_list:
            if person.name == name:
                return person
        raise AttributeError

    def call_on_each_person(self, method_to_call, **kwargs):
        print(method_to_call)
        print(kwargs)
        for person in self.person_list:
            yield method_to_call(person, **kwargs)

    def set_up(self):
        start = time.time()
        image_paths = list(paths.list_images(f"{dir_path}{self.directory}/{self.name}"))
        aligner = Aligner()

        image_paths.sort()
        total_paths = len(image_paths)

        for i, image_path in enumerate(image_paths):
            name, image_name, image, created_person = self._get_name_image_from_path(image_path, i, total_paths, aligner)

            if created_person:
                yield ('debug', f"Created new person object {name}")

            if image is None:
                yield ('warn', f"could not load image from {image_path}")
                continue

            if not self._add_embedding_to_person(image, image_path, name):
                yield ('error', f"could not embed image from {image_path}")

        print(f"\nEmbedded all of {self.person_list[-1].name}'s pictures\n")

        print(f"total time elapsed: {time.time() - start}")
        for i, person in enumerate(self.person_list):
            print(f"{i}: {person}")
        return

    def _get_name_image_from_path(self, path, position, length, aligner):
        print(f"{position + 1} of {length} images embedded")
        # The person's name should be the name of the folder they're in. (path[-2])
        name = path.split(os.path.sep)[-2]
        pic_name = path.split(os.path.sep)[-1]
        created_person = False

        if len(self.person_list) == 0:
            created_person = True
            self.add_person_to_list(Person(name))
        elif not self.has_person_named(name):
            self.add_person_to_list(Person(name))
            created_person = True
            print(f"\nembedded all of {self.person_list[-2].name}'s pictures\n")
        image = cv.imread(path)
        if image is None:
            print(f"ERROR: Could not load image from path:{path}")
        image = aligner.align_image(image)

        return name, pic_name, image, created_person

    def _add_embedding_to_person(self, image, path, name=None):
        face_blob = cv.dnn.blobFromImage(image, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        embedding_tool.setInput(face_blob)

        vec = embedding_tool.forward()
        if vec is None:
            return False

        if name is None:
            self.person_list[-1].add_image_vector(vec, path)
        else:
            self.get_person_named(name).add_image_vector(vec, path)
        return True
